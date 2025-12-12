"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

        # Safety manager
        self.safety_manager = SafetyManager({
            **config.get("safety", {}),
            "logging": config.get("logging", {}),
            "system": config.get("system", {}),
        })

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Input safety check
            safety_events: List[Dict[str, Any]] = []
            input_check = self.safety_manager.check_input_safety(query)
            if not input_check.get("safe", True):
                safety_events.append({
                    "type": "input",
                    "safe": False,
                    "violations": input_check.get("violations", []),
                })
                message = input_check.get("message") or "Query blocked by safety policy."
                return {
                    "query": query,
                    "response": message,
                    "conversation_history": [],
                    "metadata": {
                        "error": True,
                        "safety_events": safety_events,
                    },
                }

            sanitized_query = input_check.get("sanitized", query)

            # Create a fresh team for this query to avoid event loop conflicts
            self.logger.info("Creating research team for this query...")
            team = create_research_team(self.config)

            # Run the async query processing
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # If we get here, there's a running loop - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(sanitized_query, max_rounds, team)
                    ).result()
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                result = asyncio.run(self._process_query_async(sanitized_query, max_rounds, team))

            # Output safety check
            output_check = self.safety_manager.check_output_safety(result.get("response", ""), result.get("metadata", {}).get("sources", []))
            if not output_check.get("safe", True):
                safety_events.append({
                    "type": "output",
                    "safe": False,
                    "violations": output_check.get("violations", []),
                })
            result["response"] = output_check.get("response", result.get("response", ""))
            # Merge safety events
            metadata = result.get("metadata", {})
            metadata["safety_events"] = metadata.get("safety_events", []) + safety_events
            result["metadata"] = metadata

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int, team) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            team: The AutoGen team to use for this query
            
        Returns:
            Dictionary containing results
        """
        topic = self.config.get("system", {}).get("topic", "HCI research")

        task_message = f"""Research Query: {query}

    Topic Focus: {topic}
    Please keep answers concise to conserve tokens and cite sources clearly.

    Workflow:
    1. Planner: Create a research plan and finish with PLAN COMPLETE
    2. Researcher: Gather evidence from web and academic sources (use the tools) and finish with RESEARCH COMPLETE
    3. Writer: Synthesize findings into a well-cited response, end with DRAFT COMPLETE
    4. Critic: Evaluate the quality and approve or request revision"""
        
        # Run the team
        self.logger.info("Starting team.run()...")
        result = await team.run(task=task_message)
        self.logger.info(f"team.run() completed. Result type: {type(result)}")
        
        # Extract conversation history
        messages = []
        # AutoGen's TaskResult has a messages property that we need to consume
        if hasattr(result, "messages"):
            self.logger.info("Extracting messages from result...")
            # Check if it's an async iterator
            if hasattr(result.messages, "__aiter__"):
                async for message in result.messages:
                    content = getattr(message, "content", str(message))
                    # Ensure content is JSON-serializable
                    if not isinstance(content, str):
                        content = str(content)
                    msg_dict = {
                        "source": getattr(message, "source", "unknown"),
                        "content": content,
                    }
                    messages.append(msg_dict)
                    self.logger.debug(f"Message from {msg_dict['source']}: {msg_dict['content'][:100]}...")
            # Or a regular list
            elif isinstance(result.messages, list):
                for message in result.messages:
                    content = getattr(message, "content", str(message))
                    # Ensure content is JSON-serializable
                    if not isinstance(content, str):
                        content = str(content)
                    msg_dict = {
                        "source": getattr(message, "source", "unknown"),
                        "content": content,
                    }
                    messages.append(msg_dict)
                    self.logger.debug(f"Message from {msg_dict['source']}: {msg_dict['content'][:100]}...")
            else:
                self.logger.warning(f"Unexpected messages type: {type(result.messages)}")
        
        self.logger.info(f"Extracted {len(messages)} messages")
        
        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break
        
        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")
        
        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        sources: List[str] = []
        import re
        for finding in research_findings:
            # Handle both string and list content
            finding_text = finding if isinstance(finding, str) else str(finding)
            # Rough count of sources based on numbered results
            num_sources += finding_text.count("\n1.") + finding_text.count("\n2.") + finding_text.count("\n3.")

            # Extract URLs to share with UI/evaluator
            for url in re.findall(r"https?://[^\s<>'\"]+", finding_text):
                if url not in sources:
                    sources.append(url)
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": len(sources),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
                "safety_events": [],
                "sources": [{"url": s} for s in sources],
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

