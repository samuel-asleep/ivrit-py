"""
Example demonstrating how to use RunPod with persistent sessions
for low-latency transcription.
"""

import asyncio
from ivrit import load_model


async def example_persistent_session():
    """
    Example of using RunPod with persistent sessions to reduce latency.
    
    The persistent session keeps the RunPod worker warm between requests,
    eliminating cold-start latency on subsequent transcriptions.
    """
    # Create a RunPod model with persistent session enabled (default)
    model = load_model(
        engine="runpod",
        model="large-v3-turbo",
        api_key="your-api-key",
        endpoint_id="your-endpoint-id",
        use_persistent_session=True,  # This is the default
        keep_alive_interval=30.0  # Send keep-alive every 30 seconds
    )
    
    # Use async context manager for automatic cleanup
    async with model:
        # First transcription - may have cold-start latency
        print("First transcription (may have cold start)...")
        async for segment in model.transcribe_async(path="audio1.mp3", language="he"):
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        
        # Subsequent transcriptions benefit from warm worker
        # No cold-start latency!
        print("\nSecond transcription (warm worker, no cold start)...")
        async for segment in model.transcribe_async(path="audio2.mp3", language="he"):
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        
        print("\nThird transcription (still warm)...")
        async for segment in model.transcribe_async(path="audio3.mp3", language="he"):
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
    
    # Session automatically closed when exiting the context manager


async def example_without_persistent_session():
    """
    Example of using RunPod without persistent sessions (legacy behavior).
    
    Each transcription creates a new job, potentially incurring cold-start latency.
    """
    model = load_model(
        engine="runpod",
        model="large-v3-turbo",
        api_key="your-api-key",
        endpoint_id="your-endpoint-id",
        use_persistent_session=False  # Disable persistent sessions
    )
    
    # Each transcription is independent and may have cold-start latency
    print("First transcription (may have cold start)...")
    async for segment in model.transcribe_async(path="audio1.mp3", language="he"):
        print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
    
    print("\nSecond transcription (may also have cold start)...")
    async for segment in model.transcribe_async(path="audio2.mp3", language="he"):
        print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")


async def example_manual_cleanup():
    """
    Example of manually managing session lifecycle without context manager.
    """
    model = load_model(
        engine="runpod",
        model="large-v3-turbo",
        api_key="your-api-key",
        endpoint_id="your-endpoint-id"
    )
    
    try:
        # Transcribe multiple files
        async for segment in model.transcribe_async(path="audio1.mp3", language="he"):
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        
        async for segment in model.transcribe_async(path="audio2.mp3", language="he"):
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
    finally:
        # Always close to cleanup resources
        await model.close()


if __name__ == "__main__":
    # Run the persistent session example
    print("See examples above for usage patterns")
