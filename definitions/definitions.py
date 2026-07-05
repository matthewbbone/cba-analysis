from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


WageScaleFactor = Literal["occupation", "education", "seniority"]
WAGE_SCALE_FACTORS = {"occupation", "education", "seniority"}


def validate_factor_key_references(
    wage_map_name: str,
    wage_map: dict[str, object] | None,
    declared_factors: set[str],
) -> None:
    if wage_map is None:
        return

    for key in wage_map:
        for part in key.split("|"):
            if "=" not in part:
                continue
            factor = part.split("=", 1)[0].strip()
            if factor not in WAGE_SCALE_FACTORS:
                raise ValueError(
                    f"{wage_map_name} key {key!r} uses unsupported factor {factor!r}"
                )
            if factor not in declared_factors:
                raise ValueError(
                    f"{wage_map_name} key {key!r} uses undeclared factor {factor!r}"
                )


class WageRate(BaseModel):
    min: Decimal | None = Field(
        default=None,
        description=(
            "The minimum wage for this factor-value combination when the CBA "
            "states a wage range."
        ),
    )
    max: Decimal | None = Field(
        default=None,
        description=(
            "The maximum wage for this factor-value combination when the CBA "
            "states a wage range."
        ),
    )
    mid: Decimal | None = Field(
        default=None,
        description=(
            "The point wage, midpoint wage, or directly calculated midpoint for "
            "this factor-value combination when supported by the CBA."
        ),
    )
    effective_date: str | None = Field(
        default=None,
        description="The date when this wage rate takes effect, in YYYY-MM-DD format.",
    )

    @model_validator(mode="after")
    def validate_at_least_one_wage_value(self) -> Self:
        if self.min is None and self.max is None and self.mid is None:
            raise ValueError("at least one of min, max, or mid must be provided")
        return self


class WageRaise(BaseModel):
    percentage: Decimal = Field(
        description=(
            "The percentage increase for this raise, for example 0.05 for a 5% raise."
        )
    )
    effective_date: str = Field(
        description="The date when this raise takes effect, in YYYY-MM-DD format."
    )


class WageScale(BaseModel):
    factors: list[WageScaleFactor] = Field(
        default_factory=list,
        description=(
            "The factors that determine different wage rates in this scale. "
            "Omit factors that do not correspond to wage differences."
        ),
    )
    factor_values: dict[WageScaleFactor, list[str]] = Field(
        default_factory=dict,
        description=(
            "The values for the wage-determining factors. Omit factor_values when "
            "there are no wage-determining factors."
        ),
    )
    base_wages: dict[str, WageRate] = Field(
        description=(
            "Maps each unique combination of factor values to its wage rate. "
            "Keys should identify one combination using factors in order, for example "
            "'occupation=teacher|education=BA|seniority=5 years'. If no factors "
            "determine wage differences, use a single descriptive key such as "
            "'all_workers'."
        )
    )
    scheduled_raise: dict[str, WageRaise] | None = Field(
        default=None,
        description=(
            "An optional explicit schedule of percentage wage increases. Keys "
            "should describe the affected group, using factor-value combinations "
            "when raises vary by factor, for example "
            "'occupation=teacher|education=BA|seniority=5 years', or a broad key "
            "such as 'all_workers' when the same raise applies to all covered "
            "workers. If multiple raises apply to the same group, include the date "
            "in the key, for example 'all_workers_2026-04-01'. Values contain the "
            "percentage increase and effective date."
        ),
    )
    scheduled_rate: dict[str, WageRate] | None = Field(
        default=None,
        description=(
            "An optional explicit schedule that directly maps specific factor-value "
            "combinations to wage rates. Keys should identify one combination using "
            "factors in order, for example "
            "'occupation=teacher|education=BA|seniority=5 years'. If multiple "
            "scheduled rates apply to the same combination at different dates, "
            "include the date in the key. Values represent the wage rate (e.g., "
            "50000 for a $50,000 annual salary)."
        ),
    )
    cola_clause: bool = Field(
        default=False,
        description=(
            "Indicates whether the wage scale includes a cost-of-living adjustment (COLA) clause. "
        ),
    )
    time_unit: str | None = Field(
        default=None,
        description=(
            "The time unit for the wage rates, for example 'hour', 'week', "
            "'month', or 'year'. This is required when the CBA specifies wages "
            "in terms of a time unit, and a best guess based on the CBA text "
            "when the time unit is not explicitly stated. Omit when the wage "
            "rates are not based on a time unit, for example when the CBA "
            "specifies wages as a percentage of another wage."
        ),
    )

    @model_validator(mode="after")
    def validate_factor_values(self) -> Self:
        declared_factors = set(self.factors)
        if len(declared_factors) != len(self.factors):
            raise ValueError("factors must be unique")

        factor_value_keys = set(self.factor_values)
        if factor_value_keys != declared_factors:
            missing = declared_factors - factor_value_keys
            extra = factor_value_keys - declared_factors
            details = []
            if missing:
                details.append(f"missing factor_values for {sorted(missing)}")
            if extra:
                details.append(f"unexpected factor_values for {sorted(extra)}")
            raise ValueError("; ".join(details))

        for factor, values in self.factor_values.items():
            if not values:
                raise ValueError(f"factor_values for {factor!r} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"factor_values for {factor!r} must be unique")

        validate_factor_key_references(
            "base_wages",
            self.base_wages,
            declared_factors,
        )
        validate_factor_key_references(
            "scheduled_raise",
            self.scheduled_raise,
            declared_factors,
        )
        validate_factor_key_references(
            "scheduled_rate",
            self.scheduled_rate,
            declared_factors,
        )

        return self
