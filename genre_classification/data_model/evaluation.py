from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    """
    Class representing the evaluation metrics.
    """
    precision: float
    recall: float
    fscore: float
