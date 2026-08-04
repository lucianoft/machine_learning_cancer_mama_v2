from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
  """Caracteristicas de nucleos celulares (PAAF) para classificacao benigno/maligno."""

  model_config = ConfigDict(
    populate_by_name=True,
    extra="forbid",
    json_schema_extra={
      "description": (
        "Medidas extraidas de imagem de PAAF de massa mamaria. "
        "Dataset Breast Cancer Wisconsin (Diagnostic)."
      ),
    },
  )

  mean_radius: float = Field(alias='mean radius')
  mean_texture: float = Field(alias='mean texture')
  mean_perimeter: float = Field(alias='mean perimeter')
  mean_area: float = Field(alias='mean area')
  mean_smoothness: float = Field(alias='mean smoothness')
  mean_compactness: float = Field(alias='mean compactness')
  mean_concavity: float = Field(alias='mean concavity')
  mean_concave_points: float = Field(alias='mean concave points')
  mean_symmetry: float = Field(alias='mean symmetry')
  mean_fractal_dimension: float = Field(alias='mean fractal dimension')
  radius_error: float = Field(alias='radius error')
  texture_error: float = Field(alias='texture error')
  perimeter_error: float = Field(alias='perimeter error')
  area_error: float = Field(alias='area error')
  smoothness_error: float = Field(alias='smoothness error')
  compactness_error: float = Field(alias='compactness error')
  concavity_error: float = Field(alias='concavity error')
  concave_points_error: float = Field(alias='concave points error')
  symmetry_error: float = Field(alias='symmetry error')
  fractal_dimension_error: float = Field(alias='fractal dimension error')
  worst_radius: float = Field(alias='worst radius')
  worst_texture: float = Field(alias='worst texture')
  worst_perimeter: float = Field(alias='worst perimeter')
  worst_area: float = Field(alias='worst area')
  worst_smoothness: float = Field(alias='worst smoothness')
  worst_compactness: float = Field(alias='worst compactness')
  worst_concavity: float = Field(alias='worst concavity')
  worst_concave_points: float = Field(alias='worst concave points')
  worst_symmetry: float = Field(alias='worst symmetry')
  worst_fractal_dimension: float = Field(alias='worst fractal dimension')

  def to_feature_dict(self) -> dict:
    return self.model_dump(by_alias=True)


class PredictRequest(BaseModel):
  """Requisicao de predicao para uma amostra de cancer de mama."""

  patient: PatientInput


class PredictResponse(BaseModel):
  """Resultado da predicao benigno/maligno."""

  predicao: int = Field(description="0 = benigno, 1 = maligno")
  label: str = Field(description="Benigno ou Maligno")
  probabilidade_benigno: float = Field(description="Probabilidade de tumor benigno")
  probabilidade_maligno: float = Field(description="Probabilidade de tumor maligno")


class ExplainResponse(PredictResponse):
  """Resultado da predicao + explicacao em linguagem natural gerada pela LLM."""

  explicacao: str = Field(description="Explicacao em linguagem natural pro profissional de saude")


class HealthResponse(BaseModel):
  status: str
  model_name: str | None = Field(default=None, description="Modelo de ML em uso")


class MetadataResponse(BaseModel):
  """Metadados do modelo — cancer de mama (Wisconsin Diagnostic)."""

  model_name: str
  target_column: str = Field(description="Coluna alvo: diagnosis (0 benigno, 1 maligno)")
  target_labels: dict[int, str]
  feature_columns: list[str]
  dataset: str = Field(description="breast_cancer_wisconsin.csv")
