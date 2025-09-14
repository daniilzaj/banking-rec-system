from pydantic import BaseModel, Field
from typing import List, Optional
import datetime

# --- Модели для входных данных ---
# Эти модели описывают структуру наших CSV файлов.
# Они не будут использоваться напрямую в API, но полезны для структурирования кода.

class Client(BaseModel):
    client_code: str
    name: str
    status: str
    age: int
    city: str
    avg_monthly_balance_KZT: float = Field(..., alias='avg_monthly_balance_KZT')

class Transaction(BaseModel):
    client_code: str
    date: datetime.date
    category: str
    amount: float
    currency: str

class Transfer(BaseModel):
    client_code: str
    date: datetime.date
    type: str
    direction: str
    amount: float
    currency: str


# --- Модели для API ---
# Эти модели будут использоваться для ответа нашего API.

class ProductRecommendation(BaseModel):
    """
    Описывает одну рекомендацию для клиента.
    """
    client_code: str
    product_name: str
    benefit: float = Field(description="Рассчитанная выгода для клиента в KZT или %")
    push_notification_text: str

class RecommendationResponse(BaseModel):
    """
    Описывает полный ответ нашего API - список рекомендаций для всех клиентов.
    """
    recommendations: List[ProductRecommendation]