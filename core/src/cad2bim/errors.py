"""사용자에게 그대로 보여 줄 수 있는 입력 오류."""


class Cad2BimError(Exception):
    """처리를 멈춰야 하는 입력 문제. 메시지는 사용자가 할 일을 포함한다."""


class ConverterNotAvailableError(Cad2BimError):
    """DWG 변환기를 찾을 수 없음."""


class EmptyDrawingError(Cad2BimError):
    """배치된 요소가 하나도 없음. 변환기가 내용을 잃었을 수 있다."""


class XrefMissingError(Cad2BimError):
    """외부참조가 포함되지 않아 도면 일부가 빠져 있음."""


class UnitScaleError(Cad2BimError):
    """단위·축척을 판정할 수 없음. detail은 숫자만 담는 진단 코드라 공유해도 된다."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


class RegionAmbiguousError(Cad2BimError):
    """한 파일에 평면 후보가 여러 개라 --region 지정이 필요함."""

    def __init__(self, message: str, candidates: list[tuple[float, float, float, float]]):
        super().__init__(message)
        self.candidates = candidates  # mm 단위
        self.scale_factor = 1.0  # 도면 단위 → mm. --region은 도면 단위로 받는다
