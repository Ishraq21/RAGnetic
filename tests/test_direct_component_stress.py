#!/usr/bin/env python3
"""
RAGnetic Direct Component Stress Test
=====================================

Direct stress testing of components without database dependencies.
PUSH EVERY SYSTEM TO ITS BREAKING POINT!
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import string
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import uuid
import psutil

# Add RAGnetic to path
sys.path.insert(0, '/Users/ishraq21/ragnetic')

from app.agents.agent_graph import get_agent_workflow
from app.pipelines.embed import _get_chunks_from_documents, _generate_chunk_id
from app.schemas.agent import AgentConfig, VectorStoreConfig, ChunkingConfig, DataSource
from langchain_core.documents import Document as LangChainDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DirectComponentStress")

class DirectStressTester:
    """Direct component stress testing without external dependencies"""
    
    @staticmethod
    def generate_stress_content(size_mb: int = 1) -> str:
        """Generate varied content for stress testing"""
        chars_per_mb = 1024 * 1024
        total_chars = size_mb * chars_per_mb
        
        content_types = [
            lambda: ''.join(random.choices(string.ascii_letters + string.digits + ' \n\t.,;:!?()-[]{}', k=8192)),
            lambda: f"Technical documentation section {random.randint(1000, 9999)}: " + "complex analysis " * 1000,
            lambda: f"Research data: {{'id': {random.randint(1, 9999)}, 'data': '{''.join(random.choices(string.ascii_letters, k=4000))}'}},",
            lambda: f"Algorithm implementation:\nfor i in range({random.randint(100, 999)}):\n    process(data_{random.randint(1, 99)})\n" * 100
        ]
        
        content_parts = []
        for i in range(0, total_chars, 8192):
            chunk_size = min(8192, total_chars - i)
            content_type = random.choice(content_types)
            chunk = content_type()[:chunk_size]
            content_parts.append(chunk)
        
        return ''.join(content_parts)
    
    @staticmethod
    def create_test_documents(num_docs: int, doc_size_mb: float = 0.5) -> List[LangChainDocument]:
        """Create test documents for stress testing"""
        documents = []
        
        for i in range(num_docs):
            content = DirectStressTester.generate_stress_content(int(doc_size_mb))
            
            doc = LangChainDocument(
                page_content=f"STRESS TEST DOCUMENT {i}\n{'='*50}\n{content}",
                metadata={
                    'doc_name': f'stress_doc_{i:04d}.txt',
                    'source_name': f'stress_source_{i}',
                    'chunk_index': 0,
                    'page_number': 1,
                    'source_config': {'path': f'/stress/path/doc_{i}.txt'}
                }
            )
            
            documents.append(doc)
            
            if i % 100 == 0 and i > 0:
                logger.info(f"Created {i}/{num_docs} stress test documents")
        
        return documents

class TestDirectAgentGraphStress:
    """Direct stress testing of agent_graph without database"""
    
    def test_extreme_agent_concurrency_direct(self):
        """Test agent graph concurrency directly - MAXIMUM DESTRUCTION"""
        logger.info("=== DIRECT AGENT GRAPH CONCURRENCY DESTRUCTION ===")
        
        # Create simple agent config for testing
        agent_config = AgentConfig(
            name="direct_stress_agent",
            persona_prompt="You are a stress testing agent.",
            sources=[],
            tools=[],  # No tools to avoid database dependencies
            llm_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        )
        
        # Get agent workflow
        workflow = get_agent_workflow(tools=[])
        runnable = workflow.compile()
        
        # Test increasing concurrency levels
        concurrency_results = []
        
        for concurrent_agents in [5, 10, 25, 50, 100]:
            logger.info(f"DIRECT CONCURRENCY TEST: {concurrent_agents} agents - DESTRUCTION IMMINENT")
            
            async def direct_agent_task(task_id: int):
                try:
                    start_time = time.perf_counter()
                    
                    agent_state = {
                        "messages": [{"role": "human", "content": f"Execute stress test task {task_id} with maximum processing"}],
                        "tool_calls": [],
                        "request_id": f"direct_stress_{task_id}",
                        "agent_name": f"direct_destroyer_{task_id}",
                        "agent_config": agent_config,
                        "retrieval_time_s": 0.0,
                        "generation_time_s": 0.0,
                        "total_duration_s": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "retrieved_chunk_ids": [],
                        "temp_document_ids": [],
                        "retrieved_documents_meta_for_citation": [],
                        "citations": []
                    }
                    
                    config = {
                        "configurable": {
                            "thread_id": f"direct_thread_{task_id}",
                            "user_id": 1,
                            "session_id": f"direct_session_{task_id}",
                            "db_session": None,
                            "tools": [],
                            "request_id": f"direct_stress_{task_id}",
                            "agent_name": f"direct_destroyer_{task_id}"
                        }
                    }
                    
                    result = await runnable.ainvoke(agent_state, config)
                    end_time = time.perf_counter()
                    
                    return {
                        'task_id': task_id,
                        'execution_time': end_time - start_time,
                        'success': True,
                        'has_messages': bool(result.get('messages'))
                    }
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    return {
                        'task_id': task_id,
                        'execution_time': end_time - start_time,
                        'error': str(e),
                        'success': False
                    }
            
            async def run_concurrency_test():
                destruction_start = time.perf_counter()
                initial_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                # Launch concurrent destruction
                tasks = [direct_agent_task(i) for i in range(concurrent_agents)]
                
                try:
                    # Set timeout to prevent hanging
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=120
                    )
                    
                    destruction_time = time.perf_counter() - destruction_start
                    final_memory = psutil.virtual_memory().used / (1024 * 1024)
                    
                    successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
                    
                    test_result = {
                        'concurrent_agents': concurrent_agents,
                        'successful': successful,
                        'failed': concurrent_agents - successful,
                        'success_rate': successful / concurrent_agents,
                        'total_time': destruction_time,
                        'throughput': successful / destruction_time if destruction_time > 0 else 0,
                        'memory_increase_mb': final_memory - initial_memory
                    }
                    
                    concurrency_results.append(test_result)
                    
                    logger.info(f"Direct concurrency {concurrent_agents}: {successful}/{concurrent_agents} successful ({test_result['success_rate']:.1%}), {test_result['throughput']:.2f} agents/sec")
                    
                    return test_result
                    
                except asyncio.TimeoutError:
                    logger.error(f"TIMEOUT at {concurrent_agents} concurrent agents - SYSTEM OVERWHELMED")
                    return {'concurrent_agents': concurrent_agents, 'status': 'timeout'}
            
            try:
                result = asyncio.run(run_concurrency_test())
                
                # If success rate drops too low, we've found the breaking point
                if result.get('success_rate', 0) < 0.5:
                    logger.warning(f"BREAKING POINT REACHED at {concurrent_agents} agents")
                    break
                    
            except Exception as e:
                logger.error(f"Concurrency test failed at {concurrent_agents}: {e}")
                break
        
        return concurrency_results

class TestDirectEmbeddingStress:
    """Direct stress testing of embedding components"""
    
    def test_chunking_stress_direct(self):
        """Test document chunking under extreme load"""
        logger.info("=== DIRECT CHUNKING STRESS TEST ===")
        
        chunking_results = []
        
        # Test increasing document loads
        for num_docs in [50, 100, 500, 1000]:
            logger.info(f"CHUNKING STRESS TEST: {num_docs} documents - OBLITERATION INCOMING")
            
            start_time = time.perf_counter()
            initial_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            try:
                # Create massive document load
                documents = DirectStressTester.create_test_documents(num_docs, 0.5)  # 0.5MB each
                
                # Create chunking config for stress
                chunking_config = ChunkingConfig(
                    mode="default",
                    chunk_size=2048,
                    chunk_overlap=200
                )
                
                # ATTEMPT EXTREME CHUNKING
                chunks = _get_chunks_from_documents(
                    documents=documents,
                    chunking_config=chunking_config,
                    embedding_model_name="text-embedding-3-small",
                    reproducible_ids=False
                )
                
                end_time = time.perf_counter()
                final_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                processing_time = end_time - start_time
                memory_increase = final_memory - initial_memory
                
                result = {
                    'num_documents': num_docs,
                    'num_chunks': len(chunks) if chunks else 0,
                    'processing_time': processing_time,
                    'memory_increase_mb': memory_increase,
                    'docs_per_second': num_docs / processing_time if processing_time > 0 else 0,
                    'chunks_per_second': len(chunks) / processing_time if processing_time > 0 and chunks else 0,
                    'success': True
                }
                
                chunking_results.append(result)
                
                logger.info(f"Chunking survived {num_docs} docs -> {len(chunks) if chunks else 0} chunks in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"CHUNKING OBLITERATED at {num_docs} documents: {e}")
                chunking_results.append({
                    'num_documents': num_docs,
                    'error': str(e),
                    'success': False
                })
                break
        
        return chunking_results
    
    def test_chunk_id_generation_stress(self):
        """Test chunk ID generation under extreme load"""
        logger.info("=== CHUNK ID GENERATION STRESS TEST ===")
        
        # Test massive chunk ID generation
        generation_results = []
        
        for num_chunks in [1000, 10000, 50000, 100000]:
            logger.info(f"CHUNK ID GENERATION: {num_chunks} IDs - MAXIMUM DESTRUCTION")
            
            start_time = time.perf_counter()
            
            try:
                chunk_ids = []
                for i in range(num_chunks):
                    content = f"Chunk content {i} with destruction data"
                    source_context = f"stress_source_{i % 100}"
                    
                    chunk_id = _generate_chunk_id(
                        content=content,
                        source_context=source_context,
                        reproducible=False,
                        idx=i
                    )
                    
                    chunk_ids.append(chunk_id)
                    
                    if i % 10000 == 0 and i > 0:
                        logger.info(f"Generated {i}/{num_chunks} chunk IDs")
                
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                
                # Check uniqueness
                unique_ids = len(set(chunk_ids))
                
                result = {
                    'num_chunks': num_chunks,
                    'unique_ids': unique_ids,
                    'processing_time': processing_time,
                    'ids_per_second': num_chunks / processing_time if processing_time > 0 else 0,
                    'uniqueness_rate': unique_ids / num_chunks,
                    'success': True
                }
                
                generation_results.append(result)
                
                logger.info(f"ID generation survived {num_chunks} -> {unique_ids} unique ({result['uniqueness_rate']:.3%}) in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"ID GENERATION OBLITERATED at {num_chunks}: {e}")
                generation_results.append({
                    'num_chunks': num_chunks,
                    'error': str(e),
                    'success': False
                })
                break
        
        return generation_results

class TestSystemResourceLimits:
    """Test system resource limits under extreme loads"""
    
    def test_memory_exhaustion_simulation(self):
        """Simulate memory exhaustion scenarios"""
        logger.info("=== MEMORY EXHAUSTION SIMULATION ===")
        
        memory_results = []
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Test increasing memory allocations
        for allocation_mb in [100, 500, 1000, 2000, 4000]:
            logger.info(f"MEMORY ALLOCATION TEST: {allocation_mb}MB - SYSTEM DESTRUCTION")
            
            try:
                start_time = time.perf_counter()
                
                # Simulate large data structures
                large_data = []
                target_size = allocation_mb * 1024 * 1024  # Convert to bytes
                
                # Create data in chunks to monitor progress
                chunk_size = 1024 * 1024  # 1MB chunks
                for i in range(0, target_size, chunk_size):
                    chunk = bytearray(min(chunk_size, target_size - i))
                    large_data.append(chunk)
                
                end_time = time.perf_counter()
                current_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                result = {
                    'allocation_mb': allocation_mb,
                    'actual_memory_increase': current_memory - initial_memory,
                    'allocation_time': end_time - start_time,
                    'success': True
                }
                
                memory_results.append(result)
                
                logger.info(f"Memory allocation survived {allocation_mb}MB -> {result['actual_memory_increase']:.1f}MB actual increase")
                
                # Clean up to prevent system issues
                del large_data
                
            except MemoryError as e:
                logger.error(f"MEMORY SYSTEM DESTROYED at {allocation_mb}MB: {e}")
                memory_results.append({
                    'allocation_mb': allocation_mb,
                    'error': 'MemoryError',
                    'success': False
                })
                break
            except Exception as e:
                logger.error(f"Memory allocation failed at {allocation_mb}MB: {e}")
                memory_results.append({
                    'allocation_mb': allocation_mb,
                    'error': str(e),
                    'success': False
                })
                break
        
        return memory_results
    
    def test_cpu_stress_simulation(self):
        """Simulate CPU stress scenarios"""
        logger.info("=== CPU STRESS SIMULATION ===")
        
        def cpu_intensive_task(task_id: int, duration: int = 10):
            """CPU intensive computation"""
            start_time = time.perf_counter()
            result = 0
            
            while time.perf_counter() - start_time < duration:
                # CPU-intensive operations
                for i in range(100000):
                    result += i ** 2
                    result = result % 1000000
            
            return result
        
        cpu_results = []
        
        # Test increasing CPU load
        for num_cores in [1, 2, 4, 8, 12]:
            if num_cores > psutil.cpu_count():
                break
                
            logger.info(f"CPU STRESS TEST: {num_cores} cores - MAXIMUM DESTRUCTION")
            
            start_time = time.perf_counter()
            initial_cpu = psutil.cpu_percent(interval=1)
            
            # Launch CPU intensive tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = [
                    executor.submit(cpu_intensive_task, i, 15)  # 15 second tasks
                    for i in range(num_cores)
                ]
                
                # Monitor CPU during execution
                cpu_readings = []
                monitor_start = time.perf_counter()
                
                while any(not f.done() for f in futures) and time.perf_counter() - monitor_start < 20:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_readings.append(cpu_percent)
                    logger.info(f"CPU usage during {num_cores}-core stress: {cpu_percent:.1f}%")
                
                # Wait for completion
                results = [f.result() for f in futures]
            
            end_time = time.perf_counter()
            
            result = {
                'num_cores': num_cores,
                'duration': end_time - start_time,
                'peak_cpu': max(cpu_readings) if cpu_readings else 0,
                'avg_cpu': sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0,
                'cpu_readings': cpu_readings,
                'success': True
            }
            
            cpu_results.append(result)
            
            logger.info(f"CPU stress {num_cores} cores: Peak {result['peak_cpu']:.1f}%, Avg {result['avg_cpu']:.1f}%")
        
        return cpu_results

def run_comprehensive_direct_stress_test():
    """Run comprehensive direct stress testing"""
    logger.info("="*80)
    logger.info("RAGnetic DIRECT COMPONENT STRESS TEST - MAXIMUM DESTRUCTION")
    logger.info("="*80)
    
    test_results = {
        'start_time': datetime.now(),
        'system_info': {
            'cpu_cores': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform
        },
        'test_results': {}
    }
    
    # Test 1: Agent Graph Concurrency
    logger.info("\n" + "="*60)
    logger.info("AGENT GRAPH CONCURRENCY DESTRUCTION")
    logger.info("="*60)
    
    agent_tester = TestDirectAgentGraphStress()
    test_results['test_results']['agent_concurrency'] = agent_tester.test_extreme_agent_concurrency_direct()
    
    # Test 2: Document Chunking Stress
    logger.info("\n" + "="*60)
    logger.info("DOCUMENT CHUNKING OBLITERATION")
    logger.info("="*60)
    
    embedding_tester = TestDirectEmbeddingStress()
    test_results['test_results']['chunking_stress'] = embedding_tester.test_chunking_stress_direct()
    test_results['test_results']['chunk_id_generation'] = embedding_tester.test_chunk_id_generation_stress()
    
    # Test 3: System Resource Limits
    logger.info("\n" + "="*60)
    logger.info("SYSTEM RESOURCE LIMIT DESTRUCTION")
    logger.info("="*60)
    
    resource_tester = TestSystemResourceLimits()
    test_results['test_results']['memory_exhaustion'] = resource_tester.test_memory_exhaustion_simulation()
    test_results['test_results']['cpu_stress'] = resource_tester.test_cpu_stress_simulation()
    
    # Generate final report
    test_results['end_time'] = datetime.now()
    duration = (test_results['end_time'] - test_results['start_time']).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("DIRECT STRESS TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    print(f" Test Duration: {duration:.1f} seconds")
    print(f" System: {test_results['system_info']['cpu_cores']} cores, {test_results['system_info']['total_memory_gb']:.1f}GB RAM")
    
    # Analyze results
    breaking_points = []
    
    for test_name, test_data in test_results['test_results'].items():
        print(f"\n {test_name.upper().replace('_', ' ')}:")
        
        if isinstance(test_data, list):
            successful_tests = sum(1 for t in test_data if t.get('success', False))
            total_tests = len(test_data)
            
            print(f"   Tests: {successful_tests}/{total_tests} successful")
            
            # Find breaking points
            for i, test in enumerate(test_data):
                if not test.get('success', True):
                    breaking_points.append(f"{test_name}: Failed at test {i+1}")
                    break
    
    if breaking_points:
        print(f"\n🚨 BREAKING POINTS DISCOVERED:")
        for bp in breaking_points:
            print(f"   • {bp}")
    else:
        print(f"\n NO BREAKING POINTS FOUND - ALL SYSTEMS SURVIVED MAXIMUM STRESS!")
    
    # Save detailed report
    report_filename = f"ragnetic_direct_stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report: {report_filename}")
    print("="*80)
    print("DIRECT STRESS TEST COMPLETED!")
    print("="*80)
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_direct_stress_test()