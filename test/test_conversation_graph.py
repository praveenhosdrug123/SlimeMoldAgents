import pytest
import uuid
import numpy as np
from collections import defaultdict
import sys  # Provides access to Python's path handling
import os   # Provides functions for interacting with the operating system


# This complex line modifies Python's import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphs.conversation_graph import ConversationDAG, Message,ConversationManager 


@pytest.fixture
def basic_dag():
    return ConversationDAG("How are you?", str(uuid.uuid4()))

@pytest.fixture
def complex_dag():
    dag = ConversationDAG("What's ML?", str(uuid.uuid4()))
    
    # Create a branching conversation:
    #                  root
    #                /      \
    #           resp1        resp2
    #            /             \
    #        follow1         follow2
    
    resp1_id = str(uuid.uuid4())
    resp2_id = str(uuid.uuid4())
    follow1_id = str(uuid.uuid4())
    follow2_id = str(uuid.uuid4())
    
    # Add two parallel responses
    dag.add_node(Message(
        node_id=resp1_id,
        content="ML is machine learning",
        agent_type="assistant1",
        timestamp="2024-01-16"
    ), dag.root.message.node_id, "response")
    
    dag.add_node(Message(
        node_id=resp2_id,
        content="It's a subset of AI",
        agent_type="assistant2",
        timestamp="2024-01-16"
    ), dag.root.message.node_id, "response")
    
    # Add follow-up responses
    dag.add_node(Message(
        node_id=follow1_id,
        content="Can you explain more?",
        agent_type="user",
        timestamp="2024-01-16"
    ), resp1_id, "follow-up")
    
    dag.add_node(Message(
        node_id=follow2_id,
        content="What other AI subsets exist?",
        agent_type="user",
        timestamp="2024-01-16"
    ), resp2_id, "follow-up")
    
    return dag, resp1_id, resp2_id, follow1_id, follow2_id

def test_message_chain(basic_dag):
    # Test a conversation chain works correctly
    response1_id = str(uuid.uuid4())
    response1 = Message(
        node_id=response1_id,
        content="I'm good, thanks!",
        agent_type="assistant",
        timestamp="2024-01-16"
    )
    
    # Add first response
    basic_dag.add_node(response1, basic_dag.root.message.node_id, "response")
    
    # Verify chain is correct
    assert len(basic_dag.root.children) == 1
    assert basic_dag.root.children[0].message.content == "I'm good, thanks!"

def test_multiple_responses(basic_dag):
    # Test handling multiple responses to same message
    responses = [
        ("It's sunny", "assistant1"),
        ("It's cloudy", "assistant2"),
        ("It's raining", "assistant3")
    ]
    
    for content, agent in responses:
        msg = Message(
            node_id=str(uuid.uuid4()),
            content=content,
            agent_type=agent,
            timestamp="2024-01-16"
        )
        basic_dag.add_node(msg, basic_dag.root.message.node_id, "response")
    
    # Verify all responses are properly linked
    assert len(basic_dag.root.children) == 3
    
    # Check each response has correct weight
    for child_id in basic_dag.root.edge_weight:
        assert basic_dag.root.edge_weight[child_id] == 0.5

def test_deep_chain(basic_dag):
    current_parent_id = basic_dag.root.message.node_id
    expected_chain = []
    
    # Create a deep chain of 5 messages
    for i in range(5):
        msg = Message(
            node_id=str(uuid.uuid4()),
            content=f"Response level {i+1}",
            agent_type=f"agent{i}",
            timestamp="2024-01-16")
        basic_dag.add_node(msg, current_parent_id, "response")
        current_parent_id = msg.node_id
        expected_chain.append(msg.content)
    
    # Verify the chain depth
    current = basic_dag.root
    actual_chain = []
    while current.children:
        current = current.children[0]
        actual_chain.append(current.message.content)
    
    assert actual_chain == expected_chain

def test_get_all_paths(complex_dag):
    dag, *_ = complex_dag
    paths = dag.get_all_paths()
    assert len(paths) == 2  # Should have 2 distinct paths

def test_get_conversation_branch(complex_dag):
    dag, resp1_id, _, follow1_id, _ = complex_dag
    branch = dag.get_branch(follow1_id)
    root_id = dag.root.message.node_id 
    expected_branch = [root_id, resp1_id, follow1_id]
    assert branch == expected_branch

def test_get_leaf_nodes(complex_dag):
    dag, _, _, follow1_id, follow2_id = complex_dag
    leaves = dag.get_leaf_nodes() 
    assert len(leaves) == 2  # follow1 and follow2
    leaf_ids = {node.message.node_id for node in leaves}
    assert leaf_ids == {follow1_id, follow2_id}

def test_get_siblings(complex_dag):
    dag, resp1_id, resp2_id, _, _ = complex_dag
    siblings = dag.get_siblings(resp1_id)
    assert len(siblings) == 1  # resp2 is sibling of resp1
    assert siblings[0].message.node_id == resp2_id

@pytest.fixture
def similar_messages_with_long_chains():
    """Fixture that creates two similar conversation chains and one different chain"""
    
    def create_chain(base_id, prefix, depth=6):  # Increased depth
        messages = []
        base_msg = Message(
            node_id=f"{base_id}_0",
            content=f"{prefix} initial message",
            agent_type="user",
            timestamp="2024-01-01"
        )
        base_msg.content_embedding = np.array([0.1, 0.9, 0.5]) if "dog" in prefix else np.array([0.8, 0.2, 0.3])
        messages.append(base_msg)
        
        # Create main chain
        for i in range(depth):
            msg = Message(
                node_id=f"{base_id}_{i+1}",
                content=f"{prefix} response level {i+1}",
                agent_type="assistant" if i % 2 == 0 else "user",
                timestamp="2024-01-01"
            )
            messages.append(msg)
            
            # Add a branch at each level
            branch_msg = Message(
                node_id=f"{base_id}_{i+1}_branch",
                content=f"{prefix} branch at level {i+1}",
                agent_type="user",
                timestamp="2024-01-01"
            )
            messages.append(branch_msg)
        
        return messages

    chain1 = create_chain("1", "The black dog")
    chain2 = create_chain("2", "The brown dog")
    chain3 = create_chain("3", "The cat")
    
    return chain1, chain2, chain3

@pytest.fixture
def conversation_manager_with_chains(similar_messages_with_long_chains):
    """Fixture that creates a ConversationManager with long conversation chains"""
    manager = ConversationManager()
    
    chain1, chain2, chain3 = similar_messages_with_long_chains
    
    # Add each chain to the manager
    for chain in [chain1, chain2, chain3]:
        graph = ConversationDAG(chain[0].content, chain[0].node_id)
        graph.root.message = chain[0]
        manager.conversation_graphs[chain[0].node_id] = graph
        
        # Add subsequent messages in chain
        current_parent_id = chain[0].node_id
        for i, msg in enumerate(chain[1:]):
            if "branch" in msg.node_id:
                # Add as branch from previous main chain message
                parent_id = f"{msg.node_id.split('_')[0]}_{i//2}"
                manager.conversation_graphs[chain[0].node_id].add_node(msg, parent_id, "branch")
            else:
                manager.conversation_graphs[chain[0].node_id].add_node(msg, current_parent_id, "response")
                current_parent_id = msg.node_id
    
    return manager

def test_merge_long_chains(conversation_manager_with_chains):
    """Test merging conversations with long chains"""
    
    # Store initial state for comparison
    initial_graphs = conversation_manager_with_chains.conversation_graphs.copy()
    initial_dog_chains = {k: v for k, v in initial_graphs.items() 
                         if "dog" in v.root.message.content}
    
    # Run merge
    conversation_manager_with_chains.merge_similar_conversations()
    
    # Find virtual node
    virtual_nodes = [node for node in conversation_manager_with_chains.conversation_graphs.keys() 
                    if node.startswith("virtual")]
    assert len(virtual_nodes) == 1, "Should create exactly one virtual node for similar dog conversations"
    virtual_id = virtual_nodes[0]
    
    virtual_graph = conversation_manager_with_chains.conversation_graphs[virtual_id]
    
    # Test 1: Verify virtual node content
    assert "black dog" in virtual_graph.root.message.content
    assert "brown dog" in virtual_graph.root.message.content
    assert "cat" not in virtual_graph.root.message.content
    
    # Test 2: Verify original dog conversations are properly linked to virtual node
    original_dog_ids = {k for k in initial_dog_chains.keys()}
    virtual_children_ids = {child.message.node_id for child in virtual_graph.root.children}
    assert original_dog_ids == virtual_children_ids, "Virtual node should link to original dog conversations"
    
    # Test 3: Verify conversation structures are preserved
    for original_id in original_dog_ids:
        original_graph = initial_graphs[original_id]
        # Check all original nodes exist in the merged state
        for node_id in original_graph.nodes:
            assert node_id in conversation_manager_with_chains.conversation_graphs[original_id].nodes
            
        # Check the chain relationships are preserved
        original_paths = original_graph.get_all_paths()
        merged_paths = conversation_manager_with_chains.conversation_graphs[original_id].get_all_paths()
        assert len(original_paths) == len(merged_paths), f"Path count mismatch for conversation {original_id}"
    
    # Test 4: Verify cat conversation remains unchanged
    cat_graph_id = next(k for k, v in initial_graphs.items() 
                       if "cat" in v.root.message.content)
    assert cat_graph_id in conversation_manager_with_chains.conversation_graphs
    assert conversation_manager_with_chains.conversation_graphs[cat_graph_id].root.message.content == \
           initial_graphs[cat_graph_id].root.message.content
    
    # Test 5: Verify weights in virtual node
    weights_sum = sum(virtual_graph.root.edge_weight.values())
    assert abs(weights_sum - 1.0) < 1e-6, "Weights should sum to 1"
    
    # Test 6: Verify embeddings
    dog_embeddings = [g.root.message.content_embedding 
                     for g in initial_dog_chains.values()]
    expected_embedding = np.mean(dog_embeddings, axis=0)
    assert np.allclose(virtual_graph.root.message.content_embedding, expected_embedding)


def test_merge_edge_cases(conversation_manager_with_chains):
    """Test edge cases in conversation merging"""
    
    # Test 1: Merge threshold sensitivity
    # Add conversations with varying degrees of similarity
    similar_msg = Message(
        node_id="edge_1",
        content="The spotted dog runs fast",
        agent_type="user",
        timestamp="2024-01-01"
    )
    similar_msg.content_embedding = np.array([0.12, 0.88, 0.49])  # Very close to other dog embeddings
    
    somewhat_similar_msg = Message(
        node_id="edge_2",
        content="The wolf hunts quickly",
        agent_type="user",
        timestamp="2024-01-01"
    )
    somewhat_similar_msg.content_embedding = np.array([0.3, 0.7, 0.4])  # Somewhat similar
    
    # Test 2: Multiple merges
    # What happens when we merge conversations multiple times?
    conversation_manager_with_chains.merge_similar_conversations()
    conversation_manager_with_chains.merge_similar_conversations()
    
    # Virtual nodes shouldn't be re-merged
    virtual_nodes = [node for node in conversation_manager_with_chains.conversation_graphs.keys() 
                    if node.startswith("virtual")]
    assert len(virtual_nodes) == 1, "Multiple merges should not create additional virtual nodes"

def test_unionfind_multiple_merges():
    manager = ConversationManager()
    
    # Create messages with very high similarity (>0.92)
    base_embedding = np.array([0.1, 0.9, 0.5])
    base_embedding = base_embedding / np.linalg.norm(base_embedding)  # normalize
    
    msg1 = Message(
        node_id="conv1",
        content="The black dog runs",
        agent_type="user",
        timestamp="2024-01-01"
    )
    msg1.content_embedding = base_embedding
    
    msg2 = Message(
        node_id="conv2",
        content="The brown dog jumps",
        agent_type="user",
        timestamp="2024-01-01"
    )
    # Very small deviation to keep similarity > 0.92
    msg2.content_embedding = base_embedding + np.array([0, 0, 0.01])
    msg2.content_embedding = msg2.content_embedding / np.linalg.norm(msg2.content_embedding)
    
    # Add initial conversations
    for msg in [msg1, msg2]:
        graph = ConversationDAG(msg.content, msg.node_id)
        graph.root.message = msg
        manager.conversation_graphs[msg.node_id] = graph
    
    # First merge
    manager.merge_similar_conversations()
    
    # Verify first merge
    virtual_nodes = [k for k in manager.conversation_graphs.keys() if k.startswith("virtual")]
    assert len(virtual_nodes) == 1, "Should create one virtual node"
    first_virtual = manager.conversation_graphs[virtual_nodes[0]]
    assert len(first_virtual.root.children) == 2, "Virtual node should have 2 children"
    
    # Add new similar conversation
    msg3 = Message(
        node_id="conv3",
        content="The spotted dog plays",
        agent_type="user",
        timestamp="2024-01-01"
    )
    msg3.content_embedding = base_embedding + np.array([0, 0, 0.02])
    msg3.content_embedding = msg3.content_embedding / np.linalg.norm(msg3.content_embedding)
    
    graph = ConversationDAG(msg3.content, msg3.node_id)
    graph.root.message = msg3
    manager.conversation_graphs[msg3.node_id] = graph
    
    # Merge with new conversation
    manager.merge_similar_conversations()
    
    # Verify structure after adding new conversation
    virtual_nodes = [k for k in manager.conversation_graphs.keys() if k.startswith("virtual")]
    assert len(virtual_nodes) == 1, "Should still have one virtual node"
    updated_virtual = manager.conversation_graphs[virtual_nodes[0]]
    assert len(updated_virtual.root.children) == 3, "Virtual node should now have 3 children"
    
    # Add dissimilar conversation with very different embedding
    msg4 = Message(
        node_id="conv4",
        content="The cat sleeps",
        agent_type="user",
        timestamp="2024-01-01"
    )
    msg4.content_embedding = np.array([0.8, -0.2, 0.3])
    msg4.content_embedding = msg4.content_embedding / np.linalg.norm(msg4.content_embedding)
    
    graph = ConversationDAG(msg4.content, msg4.node_id)
    graph.root.message = msg4
    manager.conversation_graphs[msg4.node_id] = graph
    
    # Add another conversation similar to msg4
    msg5 = Message(
        node_id="conv5",
        content="The cat naps",
        agent_type="user",
        timestamp="2024-01-01"
    )
    msg5.content_embedding = msg4.content_embedding + np.array([0, 0, 0.01])
    msg5.content_embedding = msg5.content_embedding / np.linalg.norm(msg5.content_embedding)
    
    graph = ConversationDAG(msg5.content, msg5.node_id)
    graph.root.message = msg5
    manager.conversation_graphs[msg5.node_id] = graph
    
    # Final merge
    manager.merge_similar_conversations()
    
    # Verify correct clustering
    virtual_nodes = [k for k in manager.conversation_graphs.keys() if k.startswith("virtual")]
    assert len(virtual_nodes) == 2, "Should now have two virtual nodes"
    
    # Find dog and cat clusters
    dog_cluster = None
    cat_cluster = None
    for vnode in virtual_nodes:
        vgraph = manager.conversation_graphs[vnode]
        if len(vgraph.root.children) == 3:
            dog_cluster = vgraph
        else:
            cat_cluster = vgraph
    
    assert dog_cluster is not None, "Dog cluster should exist"
    assert cat_cluster is not None, "Cat cluster should exist"
    assert len(cat_cluster.root.children) == 2, "Cat cluster should have 2 children"
