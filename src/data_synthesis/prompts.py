"""Propmt for synthesizing new NLI, scoring, comparison, reasoning summerize, confidence_based judge."""
from typing import Text, List, Dict, Any, Optional
import re
import json
import asyncio
import random
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from langchain_interface.steps.probability_prediction_step import ProbabilityPredictionStep
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_interface.models import ChatOpenAIWithBatchAPI, BatchedAPIConfig


class ProbsCompareParser(BaseOutputParser[str]):
    """Custom output parser for comparing passages"""
    def parse(self, text: str) -> Optional[str]:
        """
        Parse the LLM output to extract the comparative result
        
        Args:
            text (str): Raw LLM output text
        
        Returns:
            Optional[str]: Parsed result (Passage A, Passage B, or None)
        """
        res = {"result": None, "reasoning": text}
        match = re.search(r'\\boxed\{(Passage A|Passage B|None)\}', text)
            
        if match:
            result = match.group(1).strip()
            res["result"] = result
            return res

        return res

class SynthesisParser(BaseOutputParser[str]):
    def parse(self, text: Text) -> Text:
        res = re.sub(r"```json\n|```", "", text).strip()
        try:
            res = json.loads(res)
        except:
            res = {}
        return res 
    
    @property
    def _type(self) -> str:
        return "synthesis"

class ProbExtractParser(BaseOutputParser[str]):
    def parse(self, text: Text) -> Text:

        result = {"reasoning": text, "probability": None}
        patterns = [
             r'\\boxed{[{]?([\d\.,]+)[}]?}',
            r"```([\d.]+)```",  # Triple quote format
            r"boxed\{([\d.]+)\}",  # LaTeX boxed format
            r"```([\d\.]+)\s*```", # eg ```0.40\n```
            r"```.*?(\d+\.\d+)\s*```" # eg ```probability: 0.40\n```
        ]
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        for pattern in patterns:
            match = re.findall(pattern, text, re.IGNORECASE)
            if match:
                try:
                    prob = float(match[-1])
                    # Validate probability is in [0,1]
                    if 0 <= prob <= 1:
                        result["probability"] = prob
                        return result
                except :
                    continue
        return result
    @property
    def _type(self) -> str:
        return "probability"

class ProbabilityJudgeParser(BaseOutputParser[str]):

    def parse(self, text: Text) -> Text:
        result = {"probability": None,"reasoning": text}
        pattern = r'\\boxed\s*{+\s*([\d\.,\s]+)\s*}+'  

        match = re.findall(pattern, text)
        if match:
            numbers = match[-1]  # Extract the content inside \boxed{}
            parsed_numbers = [float(num) for num in numbers.split(',')]  # Convert to floats
            result["probability"] = parsed_numbers
            return result
        else:
            return result
        
    @property
    def _type(self) -> str:
        return "probability"
    
class ConfidenceJudgeParser(BaseOutputParser[str]):

    def parse(self, text: Text) -> Text:
        result = {"confidence": None,"reasoning": text}
        pattern = r'\\boxed\s*{+\s*([\d\.,\s]+)\s*}+'  

        match = re.findall(pattern, text)
        if match:
            numbers = match[-1]  # Extract the content inside \boxed{}
            parsed_numbers = [float(num) for num in numbers.split(',')]  # Convert to floats
            result["confidence"] = parsed_numbers
            return result
        else:
            return result
        
    @property
    def _type(self) -> str:
        return "confidence"

class SummaryParser(BaseOutputParser[str]):
    def parse(self, text: Text) -> Text:
        return {"reasoning": text}
    
    @property
    def _type(self) -> str:
        return "summary"
    
class ReasoningSummarizer():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.vllm_cache.db"))
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen2.5-32B-Instruct",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._summary_prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant good at reasoning summary"),
             ("human", """
Simply **summarize** the reasoning process and final estimated probability (a float between 0 and 1). 
Include key factors, assumptions, and influences on the estimated probability, dismiss repetitive information.

Reasoning: {reasoning}

Begin your summary with "Reasoning:" and end with "Probability:".
""")]
)

    def _truncate_reasoning(self, reasonings: list[str], max_tokens=4000) -> list[str]:
        """Truncate reasoning text if it exceeds max_tokens"""
        max_chars = max_tokens * 4  # Rough estimation: 1 token ≈ 4 chars
        return ["" if not r else 
                (r[:max_chars] + "... [truncated]" if len(r) > max_chars else r)
                for r in reasonings]

    def get_result(self, data: List[str]) -> List[Dict[str, Any]]:
        """Get all reasoning texts and summarize them."""
        calling_chain = self._summary_prompt_template | self._llm | SummaryParser()
        reasoning_texts = self._truncate_reasoning(data)
        results = []
        for result in tqdm(calling_chain.batch(inputs=[{"reasoning": r} for r in reasoning_texts],
            config=self._runnable_config), total=len(reasoning_texts), desc="Summarizing reasoning"):
            results.append({"reasoning": result["reasoning"]})
        return results        

class ProbabilityJudge():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.vllm_cache.db"))
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        # self._llm = ChatOpenAIWithBatchAPI(
        #             temperature=0,
        #             model="gpt-4o",
        #             max_tokens=None,
        #             verbose=True,
        #             top_p=0.98
        #         )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            min_p=0.1,
            max_tokens=12800 ,# Adjust as needed for long context handling
        )        
        self._confidence_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant good at probabilistic reasoning in the real world setting"),
                ("human", """
Define the Random Variables:
- Let \( H \) represent the hypothesis: {hypothesis}
- Let \( P \) represent the premise: {premise}

Below are the estimation of conditional probability P(H∣P) based on the given premise and hypothesis from other agents using probabilistic reasoning and real-world knowledge.
Your task is to assign a confidence score (ranging from 0 to 1) to each agent's response.
                 
1. {reasoning_1}
              
2. {reasoning_2}
              
3. {reasoning_3}

4. {reasoning_4}

Important Considerations: 
- Think like a human—go beyond literal semantics by considering context, common sense, and real-world knowledge.
- Related premise and hypothesis do not necessarily cause high probability.
- Assign higher confidence to assumptions that are more commonly observed and reasoning processes that are logically sound, fully justified.
- If the premise and hypothesis both refer to an entity using an indefinite noun phrase (e.g., 'a person'), assume they refer to the same entity unless there is clear evidence suggesting otherwise.
- Errors may arise from disagreements in reasoning, differing assumptions, logical mistakes, overemphasis on corner cases, underconfidence or overconfidence, and inaccuracies in estimating P(H|P). Do not blindly trust other agents' probability estimation.

Here are some examples of probability extimation:
1. hypothesis: "Three brothers pound on some drums",
   premise: "Three men dressed in white shirts and white hats, (two with baseball caps, the leader with a white construction helmet), pounding sticks on steel and plastic drums.", 
   probability: 0.00000027,       
2. hypothesis: "There is a rock currently skipping down a pond."
   premise: "A young african boy skipping rocks."
   probability: 0.058
3. hypothesis: "The man is walking into a room."
   premise: "A man is standing in the doorway of a building."
   probability: 0.2639
4. hypothesis: "People are rollerblading for something to do."
   premise: "At least six individuals are on a team wearing helmets and knee pads while rollerblading around a skating rink."
   probability: 0.5
5. hypothesis: "A brown dog is outside and it's snowing"
   premise: "A brown dog plays in a deep pile of snow."
   probability: 0.7342
6. hypothesis: "Two girls attend a convention."
   premise: "Two girls in a crowd are dressed up, one as the cartoon character Wall-E."
   probability: 0.94
7. hypothesis: "Some kids splash in the water and interact with each other."
   premise: "many children play in the water."
   probability: 0.99
              
Output Format:
- The confidence score for other agents should be a decimal value between 0 and 1, formatted as: \\boxed{{confidence1, confidence2,confidence3,confidence4}}
- Example output: \\boxed{{0.1,0.5,0.8,0.2}}"""),
("ai", "")])
        self._likert_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant good at probabilistic reasoning in the real world setting"),
             ("human", """
Define the Random Variables:
- Let \( H \) represent the hypothesis: {hypothesis}
- Let \( P \) represent the premise: {premise}
              
Task:
Below are estimations of the conditional probability P(H∣P) based on the given premise and hypothesis, generated by other agents using probabilistic reasoning and real-world knowledge.
Your task is to assess each agent's response and assign a confidence rating on a 5-point Likert scale, based on how plausible and well-justified the estimation seems. Use the following scale:
1  Very Low Confidence: The response is implausible, poorly justified, or inconsistent with the premise or real-world knowledge.
2 - Low Confidence: The response is somewhat doubtful or lacks sufficient justification.
3 - Moderate Confidence: The response is plausible and reasonably justified but not clearly strong.
4 - High Confidence: The response is well-justified and aligns well with the premise and real-world expectations.
5 - Very High Confidence: The response is highly plausible, thoroughly justified, and strongly consistent with the premise and real-world knowledge.

Here is the reasoning process for each agent:
1. {reasoning_1}
              
2. {reasoning_2}
              
3. {reasoning_3}

4. {reasoning_4}

Important Considerations: 
- Related premise and hypothesis do not necessarily cause high probability.
- Different assumptions are acceptable. Assign higher confidence to assumptions that are more commonly observed and reasoning processes that are logically sound, fully justified.
- If the premise and hypothesis both refer to an entity using an indefinite noun phrase (e.g., 'a person'), assume they refer to the same entity unless there is clear evidence suggesting otherwise.
- Errors may arise from disagreements in reasoning, differing assumptions, logical mistakes, overemphasis on corner cases, underconfidence or overconfidence, and inaccuracies in estimating P(H|P).
- Do not blindly trust other agents' probability estimation, simply average them or take the majority without thinking to get the final probability

Here are some examples of probability extimation:
1. hypothesis: "Three brothers pound on some drums",
   premise: "Three men dressed in white shirts and white hats, (two with baseball caps, the leader with a white construction helmet), pounding sticks on steel and plastic drums.", 
   probability: 0.00000027,       
2. hypothesis: "There is a rock currently skipping down a pond."
   premise: "A young african boy skipping rocks."
   probability: 0.058
3. hypothesis: "The man is walking into a room."
   premise: "A man is standing in the doorway of a building."
   probability: 0.2639
4. hypothesis: "People are rollerblading for something to do."
   premise: "At least six individuals are on a team wearing helmets and knee pads while rollerblading around a skating rink."
   probability: 0.5
5. hypothesis: "A brown dog is outside and it's snowing"
   premise: "A brown dog plays in a deep pile of snow."
   probability: 0.7342
6. hypothesis: "Two girls attend a convention."
   premise: "Two girls in a crowd are dressed up, one as the cartoon character Wall-E."
   probability: 0.94
7. hypothesis: "Some kids splash in the water and interact with each other."
   premise: "many children play in the water."
   probability: 0.99
              
Output Format:
- The confidence point for other agents should only be integer from 1 to 5, formatted as: \\boxed{{confidence point1, confidence point2,confidence point3,confidence point4}}
- Example output: \\boxed{{1,5,4,1}}"""),
("ai", "")
])
        self._judge_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant good at probabilistic reasoning in the real world setting"),
             ("human", """
Define the Random Variables:
- Let \( H \) represent the hypothesis: {hypothesis}
- Let \( P \) represent the premise: {premise}
              
Task:
Below are the estimation of conditional probability P(H∣P) based on the given premise and hypothesis from other agents using probabilistic reasoning and real-world knowledge.
Review them. Filter out the estimation that definatly wrong, only keep the reasonable or plausible probability in your final output. 

Here are the reasoning process for each agent:
1. {reasoning_1}
              
2. {reasoning_2}
              
3. {reasoning_3}

4. {reasoning_4}

Important Considerations: 
- Think like a human—go beyond literal semantics by considering context, common sense, and real-world knowledge.
- Related premise and hypothesis do not necessarily cause high probability.
- If the premise and hypothesis both refer to an entity using an indefinite noun phrase (e.g., 'a person'), assume they refer to the same entity unless there is clear evidence suggesting otherwise.
- Errors may arise from disagreements in reasoning, differing assumptions, logical mistakes, overemphasis on corner cases, underconfidence or overconfidence, and inaccuracies in estimating P(H|P). Do not blindly trust other agents' probability estimation.

Here are some examples:
1. hypothesis: "Three brothers pound on some drums",
   premise: "Three men dressed in white shirts and white hats, (two with baseball caps, the leader with a white construction helmet), pounding sticks on steel and plastic drums.", 
   probability: 0.00000027,       
2. hypothesis: "There is a rock currently skipping down a pond."
   premise: "A young african boy skipping rocks."
   probability: 0.058
3. hypothesis: "The man is walking into a room."
   premise: "A man is standing in the doorway of a building."
   probability: 0.2639
4. hypothesis: "People are rollerblading for something to do."
   premise: "At least six individuals are on a team wearing helmets and knee pads while rollerblading around a skating rink."
   probability: 0.5
5. hypothesis: "A brown dog is outside and it's snowing"
   premise: "A brown dog plays in a deep pile of snow."
   probability: 0.7342
6. hypothesis: "Two girls attend a convention."
   premise: "Two girls in a crowd are dressed up, one as the cartoon character Wall-E."
   probability: 0.94
7. hypothesis: "Some kids splash in the water and interact with each other."
   premise: "many children play in the water."
   probability: 0.99
              
Output Format:
- The final probability you choose should be a decimal value between 0 and 1, formatted as: \\boxed{{your_final_probability}}
- Example output: \\boxed{{0.0107,0.10}}"""),
("ai", "")
])
        

   
    def basic_judge(self, data):
        calling_chain = self._judge_prompt_template | self._llm | ProbabilityJudgeParser()
        results = []
        # outputs = calling_chain.batch(inputs=data, config=self._runnable_config)
        for i,output in tqdm(calling_chain.batch_as_completed(inputs=data, config=self._runnable_config),
                                total=len(data),desc="Judging: choose likely ones"):
            result = {
                "index": data[i]["idx"],
                "hypothesis": data[i]["hypothesis"], 
                "premise": data[i]["premise"],
            }
            # Merge with parsed output
            result.update(output)
            results.append(result)
        results.sort(key=lambda x: x["index"])
        return results
        
    def confidence_judge(self,data):
        print("confidence_judge")
        calling_chain = self._confidence_prompt_template | self._llm | ConfidenceJudgeParser()
        results = []
        for i,output in tqdm(calling_chain.batch_as_completed(inputs=data, config=self._runnable_config),
                                total=len(data),desc="Judging the reasonings confidence"):
            result = {
                "index": data[i]["idx"],
                "hypothesis": data[i]["hypothesis"], 
                "premise": data[i]["premise"],
            }
            # Merge with parsed output
            result.update(output)
            results.append(result)

        results.sort(key=lambda x: x["index"])
        return results
    
    def unli_probability_judge(self, data):
        with open("datas/UNLI/ds_qwen_test_basic_summary.jsonl", "r") as f:
            ds = [json.loads(line) for line in f]
        with open("datas/UNLI/ds_qwen_test_basic.jsonl", "r") as f:
            ds_raw = [json.loads(line) for line in f]
        with open("datas/UNLI/qwen_32b_basic_test.jsonl", "r") as f:
            qwen = [json.loads(line) for line in f]
        with open("datas/UNLI/qwq_basic_test_summary.jsonl", "r") as f:
            qwq = [json.loads(line) for line in f]
        with open("datas/UNLI/qwq_basic_test.jsonl", "r") as f:
            qwq_raw = [json.loads(line) for line in f]
        with open("datas/UNLI/llama_basic_test.jsonl","r") as f:
            llama = [json.loads(line) for line in f]
        prompts = []
        for i, d in enumerate(data):
            prompt = {"hypothesis":d["hypothesis"],
                "premise":d["premise"],
                "reasoning_1":ds[d["index"]]["reasoning"],
                "reasoning_2":qwen[d["index"]]["reasoning"],
                "reasoning_3":qwq[d["index"]]["reasoning"],
                "reasoning_4":llama[d["index"]]["reasoning"]}
            prompts.append(prompt)

        calling_chain = self._confidence_prompt_template | self._llm | ConfidenceJudgeParser()
        results = []
        for i,output in tqdm(calling_chain.batch_as_completed(inputs=prompts, config=self._runnable_config),
                                total=len(data),desc="Confidence-based judge the reasonings"):
            index = data[i]["index"]
            result = {
                "index": index,
                "hypothesis": data[i]["hypothesis"], 
                "premise": data[i]["premise"],
                "probs w/o judge": [ds_raw[index]["probability"], qwen[index]["probability"], qwq_raw[index]["probability"], llama[index]["probability"]],
                "original_label": data[i]["original_label"]
            }
            # Merge with parsed output
            result.update(output)
            results.append(result)

        return results
         
class BayesianProbExtractor():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.vllm_cache.db"))
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=None,
            verbose=True,
            top_p=0.98
        )

        # self._llm = ChatOpenAIWithBatchAPI(
        #     temperature=0,
        #     model="gpt-4o",
        #     max_tokens=None,
        #     verbose=True,
        #     top_p=0.98
        # )
        self._prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant good at Bayesian reasoning in the real world setting"),
             ("human", """
Define the Random Variables:
- Let \( H \) be the event corresponding to the hypothesis: {hypothesis}.  
- Let \( P \) be the event corresponding to the premise: {premise}.  

Your task is to Estimate the Conditional Probability \( P(H \mid P) \) Using Bayesian Reasoning and Real-World Assumptions

Step-by-Step Instructions:

1. Decompose the Probability Estimation: 
   - Identify relevant factors influencing \( H \) given \( P \), and define intermediate variables \( I \) if necessary.
     \[
     P(H \mid P) = \frac{{P(P \mid H) P(H)}}{{P(P)}}
	 \]
	 - If you find it hard to estimate each probability, you can also expand \( P(P) \) or \( P(H) \) with the law of total probability.
	 - Clearly state all reasonable assumptions made in the estimation process.

2. Provide Detailed Probability Assumptions and Perform the Calculation: 
   - Assign meaningful probabilities to each term, justifying them based on real-world reasoning.  
   - Compute \( P(H \mid P) \) step by step.

3. Refinement and Contextual Adjustments:
   - If the computed probability exceeds 1, adjust individual probabilities to ensure the result remains within the valid range \( [0,1] \), rather than truncating it.  
   - Incorporate additional dependencies or priors where appropriate.

Output Format: 
- The final probability should be a decimal value between 0 and 1, formatted as follows:  
  ```your_final_probability```  
- Example output:  
  ```0.0107```
             """)]
        )
        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=3500
        )
    
    def get_result(self, data):
        calling_chain = self._prompt_template | self._llm | ProbExtractParser()
        inputs = [{"premise": d["premise"], "hypothesis": d["hypothesis"]} for d in data]
        results = []
        for idx, outcome in tqdm(calling_chain.batch_as_completed(inputs), total=len(data), desc="Extracting Probs with bayesian reasoning"):
            result = {
                "idx": idx,
                **data[idx],
                **outcome
            }
            results.append(result)

        results.sort(key=lambda x: x["idx"])
        return results
    
class ProbsExtractorPrompt():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.vllm_cache.db") )
        
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/QwQ-32B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )

        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=3300
        )
        # this prompt is for the reasoning model like DeepSeek-R1-Distill-Qwen-32B
        self._prompt_template_basic = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant good at probabilistic reasoning in the real world setting"),
             ("human","""Given a premise and a hypothesis, evaluate the probability of the hypothesis being true based on the information provided in the premise, supplemented by world knowledge and probabilistic reasoning. Specifically:
1. Use relevant world knowledge to assess contextual factors (e.g., demographics, common practices, or statistical distributions) that may influence the likelihood of the hypothesis given the premise.
2. Perform the probabilistic reasoning to estimate the conditional probability P(Hypothesis | Premise).
3. Assign a probability score between [0, 1] that quantifies P(Hypothesis | Premise). Ensure this score reflects the strength of the connection between the premise and hypothesis based on probabilistic reasoning and world knowledge.

Premise: {premise}
Hypothesis: {hypothesis}
              
Your final probability estimate should be a value in the range \([0,1]\), as fine-grained as possible, and formatted as follows: ```your_final_probability```

For example, if the estimated probability is 0.0653, the output should be: ```0.0653```
              """)] 
        )        

        self._prompt_template_vanilla = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant"),
             ("human", """
### Question: Given the premise {premise}, how likely is it that the hypothesis {hypothesis} is true?\n\n
              
The final probability should be a value between 0 and 1, formatted as follows: ```your_final_probability```
    - Example output: ```0.0107```
             """)]
            ) 
    def get_result(self, data) -> list[str]:
        # The person never drinks tea in the morning.
        calling_chain = self._prompt_template_basic | self._llm | ProbExtractParser() 
        inputs = [{"premise": d["premise"], "hypothesis": d["hypothesis"]} for d in data]  
        results = []
        for idx, outcome in tqdm(calling_chain.batch_as_completed(inputs), total=len(data), desc="Extracting Probabilities"):
            result = {
                "idx": idx,
                **data[idx],
                **outcome
            }
            results.append(result)

        results.sort(key=lambda x: x["idx"])
        return results
        
class ProbsComparePrompt():

    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.compare_cache.db"))

        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            openai_api_key = "EMPTY",
            temperature=0,
            model="Qwen/QwQ-32B",
            # model_kwargs={"top_p": 0.98},
            max_tokens=None,
            verbose=True,
            top_p=0.98
        )

        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=3000
        )

        self._prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant"),
             ("human","Given the premise and hypothesis of two natural inference passages, determine which hypothesis is more likely based on its premise.\n"
                        
                        "Specifically:\n"
                        "1. Contextual Assessment with World Knowledge\n"
                        "Analyze each pair: Evaluate the premise and hypothesis using relevant world knowledge. "
                        "Consider contextual factors such as demographics, common practices, or statistical distributions to estimate the likelihood of the hypothesis being true.\n"
                        "State assumptions: Explicitly identify any assumptions or uncertainties introduced by missing information in the premise.\n"
                        "2. Comparison\n"
                        "Compare the likelihood of each hypothesis based on the alignment between the premise and hypothesis.\n"
                        "Justify your reasoning for why one hypothesis is more likely than the other, considering the degree of alignment and the assumptions made.\n"
                        "Tie Situation: If the likelihoods of both hypotheses are sufficiently close or indistinguishable, return a None.\n\n"    
                        "Passage A: {item1}\n"
                        "Passage B: {item2}\n\n"
                        "3. Output Format Example:\n"
                        "In your final decision, strictly output \boxed{{Passage A}}, \boxed{{Passage B}} or \boxed{{None}}")])
        
    def get_random_combinations(self, 
                                 data: List[Dict], 
                                 num_combinations: int = 5000) -> List[Dict]:
        """
        Generate random unique combinations from the dataset
        
        Args:
            data (List[Dict]): Original dataset
            num_combinations (int): Number of combinations to generate
        
        Returns:
            List[Dict]: List of random combinations
        """
        if num_combinations > len(data) * (len(data) - 1):
            num_combinations = len(data) * (len(data) - 1) // 2
        
        combinations = set()
        while len(combinations) < num_combinations:
            idx1, idx2 = random.sample(range(len(data)), 2)
            if idx1 != idx2:
                combinations.add((idx1, idx2))
        
        return [{"forward":
            {
                "item1": "premise: "+data[idx1]["premise"] + " hypothesis: "+data[idx1]["hypothesis"],
                "item2": "premise: "+data[idx2]["premise"] + " hypothesis: "+data[idx2]["hypothesis"]
            },
            "reverse":
            {
                "item1": "premise: "+data[idx2]["premise"] + " hypothesis: "+data[idx2]["hypothesis"],
                "item2": "premise: "+data[idx1]["premise"] + " hypothesis: "+data[idx1]["hypothesis"]
            },
            "idx1": idx1,
            "idx2": idx2,
            "true_relation": data[idx1]["label"] > data[idx2]["label"]
            } for idx1, idx2 in combinations
        ]
    
    def get_result(self, data: List[Dict]) -> List[Optional[str]]:
        """
        Process the dataset and get comparative results
        
        Args:
            data (List[Dict]): Dataset to process
        
        Returns:
            List[Optional[str]]: Comparative results
        """
        # Create processing chain
        parsing_chain = self._prompt_template | self._llm | ProbsCompareParser()
        
        # Generate random combinations
        input_pairs = self.get_random_combinations(data, num_combinations=3500)

        f_results = []
        inputs = [i["forward"] for i in input_pairs]
        for index,output in tqdm(parsing_chain.batch_as_completed(
            inputs,config=self._runnable_config), total=len(inputs),desc="forward compare"):
            result = {
                "index":index,
                **output
            }
            f_results.append(result)
        f_results.sort(key=lambda x: x["index"])

        r_results = []
        inputs = [i["reverse"] for i in input_pairs]
        for index,output in tqdm(parsing_chain.batch_as_completed(
            inputs,config=self._runnable_config), total=len(inputs),desc="reverse compare"):
            result = {
                "index":index,
                **output
            }
            r_results.append(result)
        r_results.sort(key=lambda x: x["index"])
        results = []
        for index, (f,r) in enumerate(zip(f_results, r_results)):

            results.append({
             "idx1": input_pairs[index]["idx1"],
             "idx2": input_pairs[index]["idx2"],
             "result": self.compare_logic(f["result"], r["result"]),
             "true_relation": input_pairs[index]["true_relation"],
             "forward": f,
             "reverse": r})
        
        return results
    
    def compare_logic(self, result1:str, result2:str) -> bool: 

        if (result1 == "Passage A" and result2 == "Passage B") :#or (result1 == "Passage A" and result2 == "None"):
            return True
        elif (result1 == "Passage B" and result2 == "Passage A") :#or (result1 == "Passage B" and result2 == "None"):
            return False
        else:
            return None

class SynthesisPrompt():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/syn_cache.db"))
        self._llm = ChatOpenAIWithBatchAPI(
            temperature=0,
            model="gpt-4o",
            # model_kwargs={"top_p": 0.98},
            max_tokens=None,
            verbose=True,
            top_p=0.98
        )

        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=350
        )
        self._prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant"),
             ("human", """Given a natural language inference passage:
{passage}
Your goal:
Rewrite the original premise and hypothesis 
Generate 2 new premises related to the passage so that the probability of NLI P(hypothesis | new premise1, new premise2) ranges from **0.05 to 0.95**.

Steps to follow:
1. Rewrite the original premise and hypothesis for clarity and precision
	- Ensure both the premise and hypothesis are clear, precise, and logically sound.
	- Removed unnecessary modal verbs (e.g., "can," "might") and hedging language (e.g., "possibly," "somewhat").
	- If needed, specify a concrete example for clarity.
2. Generate new premises that modify the likelihood of the hypothesis being inferred:
	- Ensure all generated premises are factually correct and logically consistent.
	- Here are strategies you may consider to adjust the probability of inference:
		- Alternative Explanation (Misattribution): Provide a different cause for the phenomenon, weakening or shifting inference.
		- Increase Vagueness: Make the premise more general, requiring additional inference.
		- Observer-Dependent Effects: Frame the premise in a way that makes the inference more subjective or situational.
		- Instantiation: Provide a specific example that either supports or challenges the inference.
3. Categorize Premises into Four Bins Based on Probability
    - highly likely(probability~0.9): Premises that strongly support the hypothesis but may introduce slight variation or broader interpretations.
    - moderately likely(probability~0.7): Premises that are related to the passage but are more general, potentially requiring additional context to confirm the hypothesis.
	- neutral(probability~0.5): Balances support and doubt, making the inference uncertain.
	- unlikely (probability~0.3): Premises that introduce alternative mechanisms or are only tangentially related to the hypothesis.
	- contradict(probability~0.1): Challenges the hypothesis by shifting the explanation, observer perspective, or causal factor—without introducing factual errors.
4. Format the output as a valid JSON object with the following structure:
	{{
  "premise": "Your revised premise here.",
  "hypothesis": "Your revised hypothesis here.",
  "highly likely": [premise1, premise2],
  "moderately likely": [premise1, premise2],
  "neutral": [premise1, premise2],
  "unlikely": [premise1, premise2],
  "contradict": [premise1, premise2]
	}}
5. Recheck that output statements are factually accurate and format is a valid JSON.""")]
        )
    def data_to_input(self, data):
        
        if isinstance(data, list):
            return [{"passage": f"Premise: {d['premise']} \nHypothesis: {d['hypothesis']}\n"} for d in data]
        
        return {"passage": f"Premise: {data['premises']} \nHypothesis: {data['hypothesis']}\n"}

    async def get_result(self, data) -> list[str]:
        
        calling_chain = self._prompt_template | self._llm | SynthesisParser()
        passage = self.data_to_input(data)

        if isinstance(passage, list) and len(passage) > 1:
            return await calling_chain.abatch(inputs=passage, config=self._runnable_config)
        return calling_chain.invoke(input=passage, config=self._runnable_config)

# use probability step to extract the probability
class ReasoningBasedProbExtractor():
    def __init__(self):
        set_llm_cache(SQLiteCache("src/checkpoints/.vllm_cache.db"))
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen2.5-32B-Instruct",
            max_tokens=None,
            verbose=True,
            top_p=0.98
        )

        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=1500
        )

    def get_result(self, data):
        reasoning_based_step = ProbabilityPredictionStep()
        chain = reasoning_based_step.chain_llm(self._llm)
        responses = chain.batch(inputs=data, config=self._runnable_config)
        return responses
