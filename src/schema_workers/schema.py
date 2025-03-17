"""Definition of the graph schema we are going to work with.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import List, Text, Dict, Any


@dataclass
class Node:
    name: Text
    content: Dict[Text, Any]
    
    def to_dict(self) -> Dict[Text, Any]:
        return {
            'name': self.name,
            'content': self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> 'Node':
        return Node(data['name'], data['content'])
    
    def copy(self) -> 'Node':
        # from copy import deepcopy
        return Node(self.name, deepcopy(self.content))
        
        
@dataclass
class Edge:
    source: Text
    target: Text
    content: Dict[Text, Any]
    
    @property
    def name(self) -> Text:
        return f"{self.source} -> {self.target}"
    
    @name.setter
    def name(self, value: Text):
        raise AttributeError("Cannot set name of an edge.")
    
    def to_dict(self) -> Dict[Text, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'content': self.content
        }
        
    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> 'Edge':
        return Edge(data['source'], data['target'], data['content'])
    
    def copy(self) -> 'Edge':
        # from copy import deepcopy
        return Edge(self.source, self.target, deepcopy(self.content))
    
            
@dataclass
class Schema:
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[Text, Any]
    
    def to_dict(self) -> Dict[Text, Any]:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }
    
    def copy(self) -> 'Schema':
        # from copy import deepcopy
        return Schema(
            [node.copy() for node in self.nodes],
            [edge.copy() for edge in self.edges],
            deepcopy(self.metadata)
        )
    
    def to_dict(self) -> Dict[Text, Any]:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata
        }
        
    def to_mermaid(self) -> Text:
        """Convert schema to Mermaid diagram format."""
        mermaid = ["graph LR", "    classDef candidate fill:#f96,font-size:10px"]
        
        # normalize a string so that it is normal string in mermaid
        def _normalize(s: Text) -> Text:
            # Remove special characters ""
            s = s.replace('"', '_')
            return s
        
        # Add nodes
        for node in self.nodes:
            node_id = node.name.replace(" ", "_")
            label = _normalize(node.content['content'])  # Convert content dictionary to string

            if "candidates" in node.content:
                # also process candidates into 
                for cidx, candidate in enumerate(node.content['candidates']):
                    candidate_label = _normalize(candidate['content'])
                    label += f"<br>**({cidx}) {candidate_label};**"
            mermaid.append(f'    {node_id}["{label}"]' + (":::candidate" if "candidates" in node.content else ""))
            
        # Add edges
        for edge in self.edges:
            source = edge.source.replace(" ", "_")
            target = edge.target.replace(" ", "_")
            mermaid.append(f"    {source} --> {target}")
            
        return "\n".join(mermaid)
    
    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> 'Schema':
        return Schema(
            [Node.from_dict(node) for node in data['nodes']] if 'nodes' in data else [],
            [Edge.from_dict(edge) for edge in data['edges']] if 'edges' in data else [],
            data['metadata'] if 'metadata' in data else {}
        )