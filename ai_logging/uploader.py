"""AI Log Uploader for WEEX Hackathon.

Handles automatic upload of AI decisions to WEEX for hackathon verification.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from .models import AIDecision
from .logger import AILogger, get_ai_logger


class AILogUploader:
    """Uploads AI logs to WEEX API.

    Handles batching and automatic upload of AI decisions.
    """

    def __init__(
        self,
        weex_client,  # WeexClient instance
        ai_logger: Optional[AILogger] = None,
        upload_interval: int = 30,
        batch_size: int = 10,
    ):
        """Initialize the uploader.

        Args:
            weex_client: WEEX API client
            ai_logger: AI logger instance (uses global if not provided)
            upload_interval: Seconds between upload attempts
            batch_size: Max decisions per upload batch
        """
        self.weex_client = weex_client
        self.ai_logger = ai_logger or get_ai_logger()
        self.upload_interval = upload_interval
        self.batch_size = batch_size
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._upload_count = 0
        self._error_count = 0

    async def start(self):
        """Start the automatic upload task."""
        if self._running:
            logger.warning("AILogUploader already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._upload_loop())
        logger.info(
            f"AILogUploader started (interval={self.upload_interval}s, "
            f"batch_size={self.batch_size})"
        )

    async def stop(self):
        """Stop the automatic upload task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AILogUploader stopped")

    async def _upload_loop(self):
        """Background loop for uploading AI logs."""
        while self._running:
            try:
                await self.upload_pending()
            except Exception as e:
                logger.error(f"Error in upload loop: {e}")
                self._error_count += 1

            await asyncio.sleep(self.upload_interval)

    async def upload_pending(self) -> int:
        """Upload all pending AI logs.

        Returns:
            Number of decisions uploaded
        """
        pending = await self.ai_logger.get_pending_uploads()

        if not pending:
            return 0

        uploaded = []
        for decision in pending[: self.batch_size]:
            try:
                await self._upload_decision(decision)
                uploaded.append(decision)
                self._upload_count += 1
            except Exception as e:
                logger.error(f"Failed to upload AI log: {e}")
                self._error_count += 1

        if uploaded:
            await self.ai_logger.mark_uploaded(uploaded)
            logger.info(f"Uploaded {len(uploaded)} AI logs to WEEX")

        return len(uploaded)

    async def _upload_decision(self, decision: AIDecision):
        """Upload a single AI decision.

        Args:
            decision: AIDecision to upload
        """
        await self.weex_client.upload_ai_log(
            stage=decision.stage,
            model=decision.model,
            input_data=decision.input_data,
            output_data=decision.output_data,
            explanation=decision.explanation,
            order_id=decision.order_id,
        )

    async def upload_for_order(self, order_id: int) -> int:
        """Upload all AI logs for a specific order.

        Args:
            order_id: WEEX order ID

        Returns:
            Number of decisions uploaded
        """
        decisions = await self.ai_logger.get_decisions_for_order(order_id)

        if not decisions:
            logger.warning(f"No AI decisions found for order {order_id}")
            return 0

        uploaded = 0
        for decision in decisions:
            try:
                await self._upload_decision(decision)
                uploaded += 1
            except Exception as e:
                logger.error(f"Failed to upload AI log for order {order_id}: {e}")

        logger.info(f"Uploaded {uploaded} AI logs for order {order_id}")
        return uploaded

    async def force_upload_all(self) -> int:
        """Force upload all pending logs immediately.

        Returns:
            Number of decisions uploaded
        """
        total = 0
        while True:
            uploaded = await self.upload_pending()
            if uploaded == 0:
                break
            total += uploaded
        return total

    def get_stats(self) -> Dict[str, Any]:
        """Get uploader statistics.

        Returns:
            Dictionary with upload stats
        """
        logger_stats = self.ai_logger.get_stats()
        return {
            "running": self._running,
            "upload_count": self._upload_count,
            "error_count": self._error_count,
            "upload_interval": self.upload_interval,
            "batch_size": self.batch_size,
            **logger_stats,
        }
