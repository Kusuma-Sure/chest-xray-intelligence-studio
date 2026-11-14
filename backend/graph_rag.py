"""Small transparent graph-retrieval layer; no unsupported medical evidence is invented."""
from __future__ import annotations
import json
from pathlib import Path
import networkx as nx
from .config import FINDING_REGIONS, PROJECT_ROOT

EVIDENCE_FILE=PROJECT_ROOT/'docs'/'medical_evidence.json'


def build_prediction_graph(model_name: str, positive_findings: list[str]) -> nx.DiGraph:
    graph=nx.DiGraph()
    graph.add_node('uploaded_image',kind='image')
    classifier=f'classifier:{model_name}'
    graph.add_node(classifier,kind='model',name=model_name)
    graph.add_edge('uploaded_image',classifier,relation='analyzed_by')
    for finding in positive_findings:
        fnode=f'finding:{finding}'
        graph.add_node(fnode,kind='finding',name=finding)
        graph.add_edge(classifier,fnode,relation='predicted_finding')
        region=FINDING_REGIONS.get(finding)
        if region:
            rnode=f'region:{region}'
            graph.add_node(rnode,kind='region',name=region)
            graph.add_edge(fnode,rnode,relation='coarse_region_category')
    return graph


def retrieve_context(model_name: str, positive_findings: list[str]) -> dict:
    graph=build_prediction_graph(model_name,positive_findings)
    evidence=[]
    if EVIDENCE_FILE.is_file():
        try:
            items=json.loads(EVIDENCE_FILE.read_text(encoding='utf-8'))
            evidence=[item for item in items if item.get('finding') in positive_findings and item.get('source_url')]
        except (ValueError,OSError):
            evidence=[]
    relations=[]
    for source,target,data in graph.edges(data=True):
        relations.append({'source':source,'target':target,'relation':data.get('relation')})
    return {
        'nodes':graph.number_of_nodes(),'edges':graph.number_of_edges(),
        'relations':relations,'evidence':evidence,
        'message':('Retrieved configured source evidence.' if evidence else
                   'No configured source document matched these findings. Graph relations are structural context only, not medical evidence.'),
    }
