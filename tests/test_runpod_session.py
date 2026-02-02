"""
Test RunPod session manager functionality.
"""

import asyncio
import pytest
from ivrit.audio import RunPodModel, RunPodSessionManager


class TestRunPodSession:
    """Test RunPod session management functionality"""
    
    def test_runpod_model_initialization_with_persistent_session(self):
        """Test that RunPodModel initializes with persistent session enabled by default"""
        model = RunPodModel(
            model="large-v3-turbo",
            api_key="test-key",
            endpoint_id="test-endpoint",
            core_engine="faster-whisper"
        )
        
        assert model.use_persistent_session is True
        assert model._session_manager is not None
        assert isinstance(model._session_manager, RunPodSessionManager)
        assert model._keep_alive_interval == 30.0
    
    def test_runpod_model_initialization_without_persistent_session(self):
        """Test that RunPodModel can be initialized without persistent session"""
        model = RunPodModel(
            model="large-v3-turbo",
            api_key="test-key",
            endpoint_id="test-endpoint",
            core_engine="faster-whisper",
            use_persistent_session=False
        )
        
        assert model.use_persistent_session is False
        assert model._session_manager is None
    
    def test_runpod_model_custom_keep_alive_interval(self):
        """Test that RunPodModel respects custom keep-alive interval"""
        model = RunPodModel(
            model="large-v3-turbo",
            api_key="test-key",
            endpoint_id="test-endpoint",
            core_engine="faster-whisper",
            use_persistent_session=True,
            keep_alive_interval=60.0
        )
        
        assert model._keep_alive_interval == 60.0
        assert model._session_manager.keep_alive_interval == 60.0
    
    @pytest.mark.asyncio
    async def test_session_manager_lifecycle(self):
        """Test session manager lifecycle - start and close"""
        manager = RunPodSessionManager(
            api_key="test-key",
            endpoint_id="test-endpoint",
            keep_alive_interval=1.0  # Short interval for testing
        )
        
        # Initially not active
        assert manager._active is False
        
        # Get session should activate
        await manager.get_session()
        assert manager._active is True
        assert manager._keep_alive_task is not None
        
        # Close should deactivate
        await manager.close()
        assert manager._active is False
    
    @pytest.mark.asyncio
    async def test_context_manager_support(self):
        """Test that RunPodModel supports async context manager"""
        model = RunPodModel(
            model="large-v3-turbo",
            api_key="test-key",
            endpoint_id="test-endpoint",
            core_engine="faster-whisper"
        )
        
        # Use as context manager
        async with model as m:
            assert m is model
            assert m._session_manager._active is True
        
        # Should be closed after exiting context
        assert model._session_manager._active is False
    
    @pytest.mark.asyncio
    async def test_session_manager_multiple_get_session_calls(self):
        """Test that multiple get_session calls don't create multiple tasks"""
        manager = RunPodSessionManager(
            api_key="test-key",
            endpoint_id="test-endpoint",
            keep_alive_interval=1.0
        )
        
        # First call
        await manager.get_session()
        first_task = manager._keep_alive_task
        
        # Second call
        await manager.get_session()
        second_task = manager._keep_alive_task
        
        # Should be the same task
        assert first_task is second_task
        
        await manager.close()
