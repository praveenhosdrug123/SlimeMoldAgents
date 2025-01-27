# Multi-Agent Collaborative System

A decentralized multi-agent system designed for collaborative problem-solving, featuring specialized agents working together through a graph-based communication structure. We approach the problem of reproducibility of agentic conversations through our system by breaking down chain of thought through specialized instances of AI agents. We then perform a union find on extremely similar queries in order to create virtual nodes, which store multiple pathways for similar queries. Using CrossEncoder from sentence transformers we are able to rank nodes on a level by level basis to produce favourable nodes on each level. The connection of favourable levels in each level forms the critical path in the network of conversations.

## Use Cases
1. Provide cachable solutions to commonly asked questions
2. Comparing adversarial patterns to a golden set for the initial query
3. Reinforcement Learning - utilizing conversation pathways for model optimization

## Architecture

The system consists of three main components:

1. **Chief Agent**: System orchestrator
   - Implementation details in `src/multiagent.py`
   - Core routing and coordination logic

2. **Specialized Agents**: Research, Coding, and Critic agents
   - Individual agent implementations in respective modules
   - See `src/multiagent.py` for agent interfaces

3. **Graphs**: Communication optimization
   - Graph structures in `graphs/conversation_graph.py`
   - Virtual node implementation in `src/slime_mold.py`

## Core Components

- **Base Agent Framework**: Foundation for all agent types
  - See `src/multiagent.py` for implementation
  - Includes agent state management and messaging system

- **Response System**: Manages agent interactions through Graphs
  - Graph-based routing implemented in `graphs/conversation_graph.py`
  - Message handling and routing logic

- **Solution Architecture**: Stores and retrieves solution patterns
  - Virtual node implementation in `src/slime_mold.py`
  - Union-find clustering for similar queries

- **Safety Framework**: Comprehensive security measures
  - Implementation in `src/safetyframework.py`
  - Pattern detection and content filtering

## Testing

Test implementations are available in the `tests` directory:
- `tests/test_conversation.py`: Conversation graph tests
- `tests/test_slime_mold.py`: Test fixtures and utilities

## Documentation

Detailed documentation:
- [System Overview](https://docs.google.com/document/d/1o8wuVeYq_ZeparGOs2SfNrAtvW_cDuBCtlqYCJgRtNo/edit?tab=t.kj37woj54h0x)


## Safety Features

The system includes comprehensive safety features (see `src/safetyframework.py`):
- Adversarial pattern detection
- Harmful content filtering
- Manipulation pattern recognition
- Privacy protection mechanisms

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by Physarum polycephalum behavior
- Built with networkx, LangChain and Sentence Transformers

Would you like me to add any additional sections or make any modifications?


## Implementation References
For detailed implementation of specific components, please refer to:

MultiAgent System: src/multiagent.py
Safety Framework: src/safetyframework.py
Slime Mold Algorithm: src/slime_mold.py
Conversation Graphs: graphs/conversation_graph.py

## Future Research Directions

### Mechanistic Interpretability
- Inverse Jeopardy-style reasoning for decomposing agent responses
- Analysis of internal reasoning patterns across different agent types
- Mapping and visualization of decision paths in multi-agent interactions

### Model-Based Embeddings for Syntactic Code Transformations
- Syntactic Code Understanding
- Code Equivalence Detection

### Topological Weight Construction for Query-Response Manifolds
- Direct weight optimization without gradient descent
