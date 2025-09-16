# Unit tests for cost calculation and estimation services
import pytest
from unittest.mock import Mock, patch
from app.services.cost_service import (
    estimate_training_cost,
    estimate_inference_cost_per_min,
    gpu_cost,
    get_cheapest_provider
)
from app.core.cost_calculator import calculate_cost, count_tokens


class TestCostService:
    """Test cost calculation and estimation."""
    
    def test_gpu_cost_calculation(self):
        """Test basic GPU cost calculation."""
        # Basic calculation
        cost = gpu_cost(2.0, "A100", "runpod")
        assert cost >= 0.0
        assert isinstance(cost, float)
        
        # Zero hours should be zero cost
        assert gpu_cost(0.0, "A100", "runpod") == 0.0
        
        # Negative hours should raise error or return 0
        with pytest.raises(ValueError):
            gpu_cost(-1.0, "A100", "runpod")
    
    def test_estimate_training_cost(self):
        """Test training cost estimation."""
        # Mock database query for provider pricing
        with patch('app.services.cost_service.get_provider_pricing') as mock_pricing:
            mock_pricing.return_value = 2.0  # $2/hour
            
            cost = estimate_training_cost("A100", 3.0, "runpod")
            assert cost == 6.0  # 3 hours * $2/hour
    
    def test_estimate_inference_cost_per_min(self):
        """Test inference cost estimation."""
        with patch('app.services.cost_service.get_provider_pricing') as mock_pricing:
            mock_pricing.return_value = 1.0  # $1/hour
            
            cost_per_min = estimate_inference_cost_per_min("RTX4090", "vast")
            assert cost_per_min == 1.0 / 60  # $1/hour / 60 minutes
    
    def test_get_cheapest_provider(self):
        """Test finding cheapest provider."""
        with patch('app.services.cost_service.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "runpod", "gpu_type": "A100", "cost_per_hour": 2.0},
                {"name": "coreweave", "gpu_type": "A100", "cost_per_hour": 1.8},
                {"name": "vast", "gpu_type": "A100", "cost_per_hour": 2.2}
            ]
            
            cheapest = get_cheapest_provider("A100")
            assert cheapest["name"] == "coreweave"
            assert cheapest["cost_per_hour"] == 1.8
    
    def test_get_cheapest_provider_no_available(self):
        """Test when no providers are available."""
        with patch('app.services.cost_service.get_available_providers') as mock_providers:
            mock_providers.return_value = []
            
            with pytest.raises(ValueError, match="No available providers"):
                get_cheapest_provider("H100")
    
    def test_cost_with_different_gpu_types(self):
        """Test cost calculation for different GPU types."""
        gpu_types = ["RTX4090", "A100", "H100", "RTX3090"]
        
        for gpu_type in gpu_types:
            cost = gpu_cost(1.0, gpu_type, "runpod")
            assert cost >= 0.0
            assert isinstance(cost, float)
    
    def test_cost_with_different_providers(self):
        """Test cost calculation for different providers."""
        providers = ["runpod", "coreweave", "vast"]
        
        for provider in providers:
            cost = gpu_cost(1.0, "A100", provider)
            assert cost >= 0.0
            assert isinstance(cost, float)


class TestCostCalculator:
    """Test the core cost calculator."""
    
    def test_calculate_cost_llm_only(self):
        """Test cost calculation for LLM calls only."""
        cost = calculate_cost(
            llm_model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50
        )
        assert cost > 0.0
        assert isinstance(cost, float)
    
    def test_calculate_cost_embedding_only(self):
        """Test cost calculation for embeddings only."""
        cost = calculate_cost(
            embedding_model_name="text-embedding-3-small",
            embedding_tokens=1000
        )
        assert cost > 0.0
        assert isinstance(cost, float)
    
    def test_calculate_cost_combined(self):
        """Test cost calculation for both LLM and embeddings."""
        cost = calculate_cost(
            llm_model_name="gpt-4o",
            prompt_tokens=200,
            completion_tokens=100,
            embedding_model_name="text-embedding-3-large",
            embedding_tokens=500
        )
        assert cost > 0.0
        assert isinstance(cost, float)
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown models."""
        cost = calculate_cost(
            llm_model_name="unknown-model",
            prompt_tokens=100,
            completion_tokens=50
        )
        # Should not crash, might return 0 for unknown models
        assert cost >= 0.0
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, world! This is a test."
        tokens = count_tokens(text, "gpt-4")
        assert tokens > 0
        assert isinstance(tokens, int)
        
        # Longer text should have more tokens
        longer_text = text * 10
        longer_tokens = count_tokens(longer_text, "gpt-4")
        assert longer_tokens > tokens
    
    def test_count_tokens_empty(self):
        """Test token counting with empty text."""
        tokens = count_tokens("", "gpt-4")
        assert tokens == 0
    
    def test_count_tokens_unicode(self):
        """Test token counting with unicode characters."""
        unicode_text = "Hello 世界! 🌍 Testing unicode"
        tokens = count_tokens(unicode_text, "gpt-4")
        assert tokens > 0
        assert isinstance(tokens, int)


class TestCostOptimization:
    """Test cost optimization features."""
    
    def test_cost_comparison_between_providers(self):
        """Test comparing costs between providers."""
        with patch('app.services.cost_service.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "runpod", "gpu_type": "A100", "cost_per_hour": 2.0},
                {"name": "coreweave", "gpu_type": "A100", "cost_per_hour": 1.8}
            ]
            
            runpod_cost = gpu_cost(5.0, "A100", "runpod")
            coreweave_cost = gpu_cost(5.0, "A100", "coreweave")
            
            # CoreWeave should be cheaper
            assert coreweave_cost < runpod_cost
    
    def test_cost_estimation_accuracy(self):
        """Test that cost estimations are reasonably accurate."""
        # Mock realistic pricing
        with patch('app.services.cost_service.get_provider_pricing') as mock_pricing:
            mock_pricing.return_value = 2.5  # $2.50/hour for A100
            
            # 4 hours of A100 should cost $10
            estimated = estimate_training_cost("A100", 4.0, "runpod")
            assert estimated == 10.0
            
            # Fractional hours
            estimated_half = estimate_training_cost("A100", 0.5, "runpod")
            assert estimated_half == 1.25
    
    def test_budget_validation(self):
        """Test budget validation logic."""
        from app.services.cost_service import validate_budget
        
        # Valid budget
        assert validate_budget(100.0, 50.0) == True  # $50 cost, $100 budget
        
        # Insufficient budget
        assert validate_budget(30.0, 50.0) == False  # $50 cost, $30 budget
        
        # Exact budget
        assert validate_budget(50.0, 50.0) == True  # $50 cost, $50 budget
    
    def test_cost_alerts_threshold(self):
        """Test cost alert thresholds."""
        from app.services.cost_service import should_alert_cost
        
        # Under threshold
        assert should_alert_cost(10.0, 100.0, 0.5) == False  # 10% usage, 50% threshold
        
        # Over threshold
        assert should_alert_cost(60.0, 100.0, 0.5) == True  # 60% usage, 50% threshold
        
        # At threshold
        assert should_alert_cost(50.0, 100.0, 0.5) == True  # 50% usage, 50% threshold


class TestCostEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_values(self):
        """Test handling of negative values."""
        with pytest.raises(ValueError):
            gpu_cost(-1.0, "A100", "runpod")
    
    def test_zero_values(self):
        """Test handling of zero values."""
        assert gpu_cost(0.0, "A100", "runpod") == 0.0
        
        cost = calculate_cost(
            llm_model_name="gpt-4",
            prompt_tokens=0,
            completion_tokens=0
        )
        assert cost == 0.0
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        # Should handle large numbers without overflow
        large_cost = gpu_cost(1000.0, "A100", "runpod")
        assert large_cost > 0.0
        assert large_cost < float('inf')
        
        large_token_cost = calculate_cost(
            llm_model_name="gpt-4",
            prompt_tokens=1000000,
            completion_tokens=1000000
        )
        assert large_token_cost > 0.0
        assert large_token_cost < float('inf')
    
    def test_invalid_gpu_types(self):
        """Test handling of invalid GPU types."""
        # Should either return 0 or raise appropriate error
        try:
            cost = gpu_cost(1.0, "INVALID_GPU", "runpod")
            assert cost >= 0.0  # If it returns a value, should be non-negative
        except (ValueError, KeyError):
            pass  # Expected for invalid GPU types
    
    def test_invalid_providers(self):
        """Test handling of invalid providers."""
        try:
            cost = gpu_cost(1.0, "A100", "invalid_provider")
            assert cost >= 0.0
        except (ValueError, KeyError):
            pass  # Expected for invalid providers
    
    def test_concurrent_cost_calculations(self):
        """Test thread safety of cost calculations."""
        import threading
        import concurrent.futures
        
        results = []
        
        def calculate_cost_worker():
            cost = gpu_cost(1.0, "A100", "runpod")
            results.append(cost)
        
        # Run multiple calculations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(calculate_cost_worker) for _ in range(100)]
            concurrent.futures.wait(futures)
        
        # All results should be consistent
        assert len(results) == 100
        assert all(cost >= 0.0 for cost in results)
        
        # Results should be consistent (all same value for same inputs)
        unique_results = set(results)
        assert len(unique_results) == 1  # Should all be the same


@pytest.mark.benchmark
class TestCostPerformance:
    """Performance tests for cost calculations."""
    
    def test_cost_calculation_performance(self, benchmark):
        """Benchmark cost calculation performance."""
        def cost_calc():
            return calculate_cost(
                llm_model_name="gpt-4o",
                prompt_tokens=1000,
                completion_tokens=500,
                embedding_model_name="text-embedding-3-small",
                embedding_tokens=2000
            )
        
        result = benchmark(cost_calc)
        assert result > 0.0
    
    def test_token_counting_performance(self, benchmark):
        """Benchmark token counting performance."""
        text = "This is a sample text for benchmarking token counting performance. " * 100
        
        def token_count():
            return count_tokens(text, "gpt-4")
        
        result = benchmark(token_count)
        assert result > 0
