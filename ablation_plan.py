from __future__ import annotations


CASE1_ABLATION_PLAN = [
    {
        "id": "A1",
        "name": "full_pa_ma_llms",
        "description": "Full PA-MA-LLMs with heterogeneous profiles, dual-layer agents, resource-agent feedback, and physics-aware projection.",
    },
    {
        "id": "A2",
        "name": "no_heterogeneity_profile",
        "description": "Replace heterogeneous profiles with neutral preference and resource-feedback values while keeping the same projection layer.",
    },
    {
        "id": "A3",
        "name": "parameterized_preference_agent",
        "description": "Replace LLM agents with parameterized preference agents under the same projection layer.",
    },
    {
        "id": "A4",
        "name": "no_resource_agent_feedback",
        "description": "Disable resource-agent feedback signals while keeping operator preferences and projection unchanged.",
    },
    {
        "id": "A5",
        "name": "single_layer_agent",
        "description": "Collapse the dual-layer agent structure into a single manager with the same projection layer.",
    },
]


CASE2_ABLATION_PLAN = [
    {
        "id": "C1",
        "name": "full_pa_ma_llms",
        "description": "Full PA-MA-LLMs with heterogeneous profiles, feedback memory, double auction, and physics-aware clearing.",
    },
    {
        "id": "C2",
        "name": "homogeneous_market_preferences",
        "description": "Remove heterogeneous profiles and use homogeneous market preferences.",
    },
    {
        "id": "C3",
        "name": "no_memory_feedback",
        "description": "Disable previous order-book and projection feedback in agent memory.",
    },
    {
        "id": "C4",
        "name": "rule_based_bidding_same_projection",
        "description": "Use rule-based bidding with the same double-auction and physics-aware clearing layer.",
    },
    {
        "id": "C5",
        "name": "parameterized_bidding_same_projection",
        "description": "Use parameterized bidding agents with the same double-auction and physics-aware clearing layer.",
    },
]
