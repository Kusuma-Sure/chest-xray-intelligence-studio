"""Unit tests for Biomedical Knowledge Graph and GraphRAG evidence retrieval."""
from backend.config import FINDINGS, FINDING_ANATOMY
from backend.graph_rag import (
    _KNOWLEDGE_GRAPH,
    build_inference_subgraph,
    build_strict_grounding_prompt,
    generate_factual_grounded_explanation,
    retrieve_graph_context,
)


def test_knowledge_graph_node_coverage():
    graph = _KNOWLEDGE_GRAPH.graph
    # Verify all 13 findings exist as nodes
    for finding in FINDINGS:
        assert finding in graph, f"Finding '{finding}' missing from Knowledge Graph."
        # Verify region connection
        meta = FINDING_ANATOMY[finding]
        region = meta["region"]
        assert region in graph
        assert graph.has_edge(finding, region)


def test_evidence_citations_loaded():
    records = _KNOWLEDGE_GRAPH.evidence_records
    assert len(records) >= 13, "Should have verified citations for at least all 13 CheXpert findings."
    findings_with_evidence = {r["finding"] for r in records}
    for finding in FINDINGS:
        assert finding in findings_with_evidence, f"Missing verified citation for {finding}."


def test_graphrag_retrieval_and_strict_grounding():
    positives = [
        {"finding": "Cardiomegaly", "probability": 0.84},
        {"finding": "Pleural Effusion", "probability": 0.76},
    ]
    res = retrieve_graph_context(
        model_name="Vision Transformer (ViT-B/16)",
        positive_findings=positives,
        target_finding="Cardiomegaly",
        xai_method="ViT Gradient × Activation",
    )
    assert res["nodes"] > 10
    assert len(res["evidence"]) >= 2
    assert "grounded_explanation" in res
    assert "prompt_template" in res

    grounded = res["grounded_explanation"]
    assert "Cardiomegaly" in grounded["grounded_text"]
    assert "Pleural Effusion" in grounded["grounded_text"]
    assert "Cardiac Silhouette" in grounded["anatomical_regions"]
    assert "Pleural Space" in grounded["anatomical_regions"]
    assert len(grounded["retrieved_evidence"]) >= 2
    assert "disclaimer" in grounded["grounded_text"].lower() or "disclaimer" in grounded
