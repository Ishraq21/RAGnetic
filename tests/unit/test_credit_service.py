# Unit tests for credit management and billing services
import pytest
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from app.services.credit_service import (
    get_user_credits,
    ensure_balance,
    deduct,
    top_up,
    within_limits,
    reserve_credits,
    release_credits,
    charge_reserved
)
from app.schemas.billing import UserCredits


class TestCreditService:
    """Test credit management operations."""
    
    @pytest.mark.asyncio
    async def test_get_user_credits_new_user(self, db_session):
        """Test getting credits for a new user."""
        user_id = 999  # Non-existent user
        credits = await get_user_credits(user_id)
        
        assert isinstance(credits, UserCredits)
        assert credits.balance == 0.0
        assert credits.daily_limit > 0.0
        assert credits.total_spent == 0.0
    
    @pytest.mark.asyncio
    async def test_top_up_credits(self, db_session):
        """Test topping up user credits."""
        user_id = 1
        initial_balance = 0.0
        top_up_amount = 50.0
        
        # Top up credits
        await top_up(user_id, top_up_amount)
        
        # Verify balance updated
        credits = await get_user_credits(user_id)
        assert credits.balance == initial_balance + top_up_amount
    
    @pytest.mark.asyncio
    async def test_top_up_multiple_times(self, db_session):
        """Test multiple top-ups accumulate correctly."""
        user_id = 1
        
        await top_up(user_id, 25.0)
        await top_up(user_id, 30.0)
        await top_up(user_id, 15.0)
        
        credits = await get_user_credits(user_id)
        assert credits.balance == 70.0
    
    @pytest.mark.asyncio
    async def test_deduct_credits_sufficient_balance(self, db_session):
        """Test deducting credits when sufficient balance exists."""
        user_id = 1
        
        # Top up first
        await top_up(user_id, 100.0)
        
        # Deduct credits
        await deduct(user_id, 25.0, "Test deduction")
        
        credits = await get_user_credits(user_id)
        assert credits.balance == 75.0
        assert credits.total_spent == 25.0
    
    @pytest.mark.asyncio
    async def test_deduct_credits_insufficient_balance(self, db_session):
        """Test deducting credits when insufficient balance."""
        user_id = 1
        
        # Only top up small amount
        await top_up(user_id, 10.0)
        
        # Try to deduct more than available
        with pytest.raises(ValueError, match="Insufficient credits"):
            await deduct(user_id, 50.0, "Test deduction")
    
    @pytest.mark.asyncio
    async def test_ensure_balance_sufficient(self, db_session):
        """Test ensure_balance when user has sufficient credits."""
        user_id = 1
        await top_up(user_id, 100.0)
        
        # Should not raise exception
        await ensure_balance(user_id, 50.0)
    
    @pytest.mark.asyncio
    async def test_ensure_balance_insufficient(self, db_session):
        """Test ensure_balance when user has insufficient credits."""
        user_id = 1
        await top_up(user_id, 30.0)
        
        # Should raise exception
        with pytest.raises(ValueError, match="Insufficient credits"):
            await ensure_balance(user_id, 50.0)
    
    @pytest.mark.asyncio
    async def test_within_limits_daily(self, db_session):
        """Test daily spending limits."""
        user_id = 1
        
        # Set up user with daily limit
        await top_up(user_id, 200.0)  # Plenty of balance
        
        # Mock getting user with specific daily limit
        with patch('app.services.credit_service.get_user_credits') as mock_get:
            mock_get.return_value = UserCredits(
                balance=200.0,
                daily_limit=50.0,
                total_spent=0.0
            )
            
            # Within limit
            assert await within_limits(user_id, 30.0) == True
            
            # Over limit
            assert await within_limits(user_id, 60.0) == False
    
    @pytest.mark.asyncio
    async def test_reserve_and_release_credits(self, db_session):
        """Test credit reservation system."""
        user_id = 1
        await top_up(user_id, 100.0)
        
        # Reserve credits
        await reserve_credits(user_id, 30.0, "GPU training reservation")
        
        credits = await get_user_credits(user_id)
        assert credits.balance == 70.0  # Available balance reduced
        # Reserved balance should be tracked separately
        
        # Release reservation
        await release_credits(user_id, 30.0, "Training cancelled")
        
        credits = await get_user_credits(user_id)
        assert credits.balance == 100.0  # Balance restored
    
    @pytest.mark.asyncio
    async def test_charge_reserved_credits(self, db_session):
        """Test charging reserved credits."""
        user_id = 1
        await top_up(user_id, 100.0)
        
        # Reserve credits
        await reserve_credits(user_id, 50.0, "GPU training")
        
        # Charge part of reservation (actual usage)
        await charge_reserved(user_id, 35.0, "Actual GPU usage")
        
        # Release remaining reservation
        await release_credits(user_id, 15.0, "Unused reservation")
        
        credits = await get_user_credits(user_id)
        assert credits.balance == 65.0  # 100 - 35 charged
        assert credits.total_spent == 35.0
    
    @pytest.mark.asyncio
    async def test_concurrent_credit_operations(self, db_session):
        """Test thread safety of credit operations."""
        import asyncio
        user_id = 1
        
        # Initial top-up
        await top_up(user_id, 1000.0)
        
        # Concurrent deductions
        async def deduct_worker(amount, description):
            try:
                await deduct(user_id, amount, f"Concurrent {description}")
                return True
            except ValueError:
                return False  # Insufficient funds
        
        # Run 20 concurrent deductions of $10 each
        tasks = [
            deduct_worker(10.0, f"task_{i}")
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful deductions
        successful = sum(1 for r in results if r is True)
        
        # Verify final balance
        credits = await get_user_credits(user_id)
        expected_balance = 1000.0 - (successful * 10.0)
        assert credits.balance == expected_balance
    
    @pytest.mark.asyncio
    async def test_negative_amounts_rejected(self, db_session):
        """Test that negative amounts are rejected."""
        user_id = 1
        
        with pytest.raises(ValueError, match="Amount must be positive"):
            await top_up(user_id, -10.0)
        
        with pytest.raises(ValueError, match="Amount must be positive"):
            await deduct(user_id, -5.0, "Invalid deduction")
        
        with pytest.raises(ValueError, match="Amount must be positive"):
            await reserve_credits(user_id, -20.0, "Invalid reservation")
    
    @pytest.mark.asyncio
    async def test_zero_amounts_rejected(self, db_session):
        """Test that zero amounts are rejected."""
        user_id = 1
        
        with pytest.raises(ValueError, match="Amount must be positive"):
            await top_up(user_id, 0.0)
        
        with pytest.raises(ValueError, match="Amount must be positive"):
            await deduct(user_id, 0.0, "Zero deduction")
    
    @pytest.mark.asyncio
    async def test_precision_handling(self, db_session):
        """Test handling of decimal precision."""
        user_id = 1
        
        # Top up with precise decimal
        await top_up(user_id, 33.33)
        await top_up(user_id, 66.67)
        
        credits = await get_user_credits(user_id)
        assert abs(credits.balance - 100.0) < 0.01  # Handle floating point precision
    
    @pytest.mark.asyncio
    async def test_large_amounts(self, db_session):
        """Test handling of large credit amounts."""
        user_id = 1
        large_amount = 999999.99
        
        await top_up(user_id, large_amount)
        credits = await get_user_credits(user_id)
        assert credits.balance == large_amount
        
        # Deduct large amount
        await deduct(user_id, large_amount / 2, "Large deduction")
        credits = await get_user_credits(user_id)
        assert abs(credits.balance - large_amount / 2) < 0.01


class TestCreditTransactions:
    """Test credit transaction logging."""
    
    @pytest.mark.asyncio
    async def test_transaction_logging_top_up(self, db_session):
        """Test that top-ups are logged as transactions."""
        user_id = 1
        amount = 50.0
        
        with patch('app.services.credit_service.log_transaction') as mock_log:
            await top_up(user_id, amount)
            
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == user_id  # user_id
            assert args[1] == amount  # amount
            assert args[2] == "topup"  # transaction_type
    
    @pytest.mark.asyncio
    async def test_transaction_logging_deduction(self, db_session):
        """Test that deductions are logged as transactions."""
        user_id = 1
        await top_up(user_id, 100.0)  # Setup balance
        
        with patch('app.services.credit_service.log_transaction') as mock_log:
            await deduct(user_id, 25.0, "Test deduction")
            
            mock_log.assert_called()
            # Should be called for deduction (not the initial top_up in this test)
            deduction_call = [call for call in mock_log.call_args_list 
                             if call[0][2] == "usage"][-1]
            
            assert deduction_call[0][0] == user_id
            assert deduction_call[0][1] == 25.0
            assert deduction_call[0][2] == "usage"
    
    @pytest.mark.asyncio
    async def test_get_transaction_history(self, db_session):
        """Test retrieving transaction history."""
        user_id = 1
        
        # Perform various operations
        await top_up(user_id, 100.0)
        await deduct(user_id, 25.0, "GPU training")
        await top_up(user_id, 50.0)
        
        from app.services.credit_service import get_transaction_history
        transactions = await get_transaction_history(user_id, limit=10)
        
        assert len(transactions) >= 3
        assert any(t.transaction_type == "topup" for t in transactions)
        assert any(t.transaction_type == "usage" for t in transactions)


class TestCreditLimits:
    """Test credit limit enforcement."""
    
    @pytest.mark.asyncio
    async def test_daily_limit_enforcement(self, db_session):
        """Test that daily limits are enforced."""
        user_id = 1
        
        # Mock user with $50 daily limit
        with patch('app.services.credit_service.get_user_credits') as mock_get:
            mock_get.return_value = UserCredits(
                balance=1000.0,  # Plenty of balance
                daily_limit=50.0,
                total_spent=0.0
            )
            
            # First deduction within limit
            await deduct(user_id, 30.0, "First expense")
            
            # Second deduction that would exceed daily limit
            with patch('app.services.credit_service.get_daily_spending') as mock_daily:
                mock_daily.return_value = 30.0  # Already spent $30 today
                
                with pytest.raises(ValueError, match="Daily spending limit"):
                    await deduct(user_id, 25.0, "Would exceed limit")
    
    @pytest.mark.asyncio
    async def test_update_daily_limit(self, db_session):
        """Test updating user's daily limit."""
        user_id = 1
        new_limit = 200.0
        
        from app.services.credit_service import update_daily_limit
        await update_daily_limit(user_id, new_limit)
        
        credits = await get_user_credits(user_id)
        assert credits.daily_limit == new_limit
    
    @pytest.mark.asyncio
    async def test_spending_reset_daily(self, db_session):
        """Test that daily spending tracking resets."""
        user_id = 1
        
        from app.services.credit_service import get_daily_spending
        from datetime import datetime, timedelta
        
        # Mock spending from yesterday vs today
        with patch('app.services.credit_service.get_transactions_since') as mock_transactions:
            # Yesterday's spending
            yesterday = datetime.utcnow() - timedelta(days=1)
            mock_transactions.return_value = [
                Mock(amount=30.0, created_at=yesterday)
            ]
            
            daily_spending = await get_daily_spending(user_id)
            assert daily_spending == 0.0  # Yesterday's spending shouldn't count
            
            # Today's spending
            today = datetime.utcnow()
            mock_transactions.return_value = [
                Mock(amount=20.0, created_at=today)
            ]
            
            daily_spending = await get_daily_spending(user_id)
            assert daily_spending == 20.0


class TestCreditEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_invalid_user_id(self, db_session):
        """Test operations with invalid user ID."""
        invalid_user_id = -1
        
        # Should handle gracefully or raise appropriate error
        credits = await get_user_credits(invalid_user_id)
        assert isinstance(credits, UserCredits)
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, db_session):
        """Test handling of database errors."""
        user_id = 1
        
        with patch('app.services.credit_service.db_session.execute') as mock_execute:
            mock_execute.side_effect = Exception("Database connection error")
            
            with pytest.raises(Exception):
                await top_up(user_id, 50.0)
    
    @pytest.mark.asyncio
    async def test_rollback_on_error(self, db_session):
        """Test that transactions are rolled back on error."""
        user_id = 1
        await top_up(user_id, 100.0)
        
        initial_credits = await get_user_credits(user_id)
        initial_balance = initial_credits.balance
        
        # Mock an error during deduction
        with patch('app.services.credit_service.log_transaction') as mock_log:
            mock_log.side_effect = Exception("Logging error")
            
            with pytest.raises(Exception):
                await deduct(user_id, 25.0, "Should fail")
        
        # Balance should be unchanged due to rollback
        final_credits = await get_user_credits(user_id)
        assert final_credits.balance == initial_balance
    
    @pytest.mark.asyncio
    async def test_very_small_amounts(self, db_session):
        """Test handling of very small credit amounts."""
        user_id = 1
        tiny_amount = 0.01  # 1 cent
        
        await top_up(user_id, tiny_amount)
        credits = await get_user_credits(user_id)
        assert credits.balance >= tiny_amount
        
        await deduct(user_id, tiny_amount, "Tiny deduction")
        credits = await get_user_credits(user_id)
        assert credits.balance >= 0.0


@pytest.mark.performance
class TestCreditPerformance:
    """Performance tests for credit operations."""
    
    @pytest.mark.asyncio
    async def test_credit_operation_performance(self, db_session, benchmark):
        """Benchmark credit operations."""
        user_id = 1
        
        async def credit_operations():
            await top_up(user_id, 100.0)
            await deduct(user_id, 25.0, "Test")
            return await get_user_credits(user_id)
        
        result = await benchmark(credit_operations)
        assert result.balance == 75.0
    
    @pytest.mark.asyncio
    async def test_concurrent_users_performance(self, db_session):
        """Test performance with many concurrent users."""
        import asyncio
        
        async def user_operations(user_id):
            await top_up(user_id, 50.0)
            await deduct(user_id, 10.0, f"User {user_id} operation")
            return await get_user_credits(user_id)
        
        # Test with 100 concurrent users
        start_time = asyncio.get_event_loop().time()
        
        tasks = [user_operations(i) for i in range(100, 200)]  # User IDs 100-199
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 100 users
        assert len(results) == 100
        assert all(r.balance == 40.0 for r in results)  # 50 - 10 = 40
