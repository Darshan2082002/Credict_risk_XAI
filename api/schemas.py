from pydantic import BaseModel

class CreditInput(BaseModel):
    credit_utilization_ratio_woe: float
    debt_to_income_ratio_woe: float
    employment_years_woe: float
    number_of_late_payments_12m_woe: float
    annual_income_woe: float