"""Data models for AI logging."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AIDecision(BaseModel):
    """Represents a single AI decision for logging."""

    # Required fields
    stage: str = Field(..., description="Trading stage (e.g., 'Strategy Generation')")
    model: str = Field(..., description="AI model name (e.g., 'gpt-4-turbo')")
    input_data: Dict[str, Any] = Field(..., description="Input to the AI model")
    output_data: Dict[str, Any] = Field(..., description="Output from the AI model")
    explanation: str = Field(
        ..., max_length=1000, description="Natural language explanation"
    )

    # Optional fields
    order_id: Optional[int] = Field(None, description="WEEX order ID if applicable")
    agent_name: Optional[str] = Field(None, description="Name of the agent that made decision")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")

    # Metadata
    duration_ms: Optional[int] = Field(None, description="Time taken for AI inference")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")


class TradeDecision(BaseModel):
    """Complete trade decision with all associated AI logs."""

    trade_id: str = Field(..., description="Internal trade ID")
    symbol: str = Field(..., description="Trading pair")
    side: str = Field(..., description="buy or sell")
    size: float = Field(..., description="Order size")
    price: Optional[float] = Field(None, description="Order price")

    # AI decisions at each stage
    analysis_decision: Optional[AIDecision] = Field(
        None, description="Market analysis AI decision"
    )
    risk_decision: Optional[AIDecision] = Field(
        None, description="Risk assessment AI decision"
    )
    execution_decision: Optional[AIDecision] = Field(
        None, description="Execution AI decision"
    )

    # Result
    order_id: Optional[int] = Field(None, description="WEEX order ID after placement")
    status: str = Field(default="pending", description="Trade status")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AILogBatch(BaseModel):
    """Batch of AI logs for upload."""

    decisions: list[AIDecision] = Field(default_factory=list)
    uploaded: bool = Field(default=False)
    upload_time: Optional[datetime] = None
    upload_response: Optional[Dict[str, Any]] = None
