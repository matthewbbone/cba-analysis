import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.render import RenderedPage
from pipeline.stg_01_ocr.repetition import GenerationQualityError
from pipeline.stg_01_ocr.specialized import miner


# Exact page_1.txt emitted by the 2026-08-12 native-layout smoke regression.
# Keeping it inline makes the regression reproducible outside the machine whose
# /tmp directory originally held the failed run.
_SMOKE_GIBBERISH_B64 = (
    "64O14ouy4p+n4L2Z4oyn6riR8J2Ymu+tse+wjOCsieC0j++yh+KwjuSouuGInO2WjeuQ"
    "uOG1pfCdhLnrr5zkgIDjib/wnZOO67mq4YyU766E7L2s7Kmw6oy84ai44r6n762b4Ya6"
    "77GI4Y6S77KH8J+AhO+ug++llfCdk6LwnZOe65eN4pak76aP7Lqt44e64Ky97Y2I77+9"
    "766D4aWk1JHvv73gtI/jiavssqfshJDimqfshqzwkI2EyZjthqfvuY3vpbHwkIyw4ai4"
    "762e77Cc4YOq7Ya54byt4oqm7KCw76SJ4bi08J2HnOC0hvCdk4vhtJPhiInhoLfwnZep"
    "4Ya67JeD4LKM76ab76254Yyo46uq4oeH76W/8J2TsOusq+Kaj/Cfj4/ikqDji7Ttlqzh"
    "uLvKtuOIvO+vk+GgoeKGnOG6jeGEiuSouvCdmoTrmK3hpJPvsJntk6zvrbLihqTtnZ/v"
    "v73iiL7wn4+H8J2Vuu+vrPCdkLHqv4/hgL/ru4XqppTrlK7imqPkgIA8fHBhcmF0ZXh0"
    "fD7jiabhlpPvr6nvsIzkoIDsno/sgq7hla7wnZqB7Z2t4oeH4omT4LKk67q16rSZ766w"
    "4YCn77Or77SY4aCu77CQ4pqf6quA8JCkjeKLquyWvuOiqOGKg+KwoO+yh+KunuOEjeyS"
    "r+KwkOyequG/rOGUhe2Fo/Cdlbo8fHZpc2lvbl9zdGFydHw+7K+V67Kb76S18J+FkOGh"
    "tOG8k+qzvuGopuGFreKGnOKXg+GLm+u3l+2Vp+yXnOGIvfCflIfhpIrij4zhuaDvqIPr"
    "iJrti43hjJTgtLTvrZvimoXwnYaz7YSU8JODsOC0q++/veKUpfCdlbrhraPqqpzruoPr"
    "h5zvv73vtL3ihr3ip73vsqDhpJPvr6zvs5Trl43slZDwnZms4aCh4r+77Y2/8J+Vi++y"
    "gPCdmbvhvK3wn4aG7ZW28J+Ug+Kyre+wme2DmeKVje2Xo+GVt+Cyqu+kteusvvCdmbjw"
    "koyo76Ws762n76+78J2ZvuGFpu+vrvCflIPsg5nhtYzslr7hjITgv4DisKDjh7vhupHq"
    "ppThhrrhhrvrrKDwn4WQ76ee67u04r6b65SJ66657LKu8JKEt/Cdmb/hiIzjib/suJnt"
    "iaTimoTvp4bwnZKwPHxxdWFkX3N0YXJ0fD7ivZ/iib/vv73tk6risJDhpaTwkoC477+9"
    "76WS8J+PqeOPseGQoOG4u++uk+qdm+Cws+yYhPCfj4fir4jwn4Wm4aCG4KqG762w8J2Z"
    "v+2HvPCQjYPhjYrwnYWO8J2Gs+CgjOGMveC8heqqke2WmOyOmPCflZbvv73jiLPrpILs"
    "g4/wnYSF4b6n77+94Yyo77+977+u44mr7J2Q65qA4LKM4Ymi8J2ZkuuVp+KYrO+nhuq2"
    "leKPjAo="
)


def _smoke_gibberish() -> str:
    payload = base64.b64decode(_SMOKE_GIBBERISH_B64)
    assert hashlib.sha256(payload).hexdigest() == (
        "7068c6b963d21e3832639ce063a0528a793f5454586b28b599b2ac76d1343dea"
    )
    return payload.decode("utf-8")


class AlwaysSameCompletions:
    def __init__(self, text: str, *, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text),
                    finish_reason=self.finish_reason,
                )
            ]
        )


def _make_context(args, completions) -> common.PageContext:
    return common.PageContext(
        job=common.PageJob(
            "source", "doc", Path("doc.pdf"), 1, Path("page_1.txt")
        ),
        rendered_page=RenderedPage("unused"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model_name=args.model_name,
        args=args,
        request_limiter=common.RequestLimiter(args.max_inflight_requests),
        repetition_policy=miner.SPEC.repetition_policy(args),
    )


class MinerGibberishDetectorTests(unittest.TestCase):
    def test_policy_uses_miner_output_quality_callback(self) -> None:
        policy = miner.repetition_policy(miner.parse_args([]))

        self.assertIs(policy.output_rejection_reason, miner.gibberish_reason)

    def test_rejects_printable_ascii_ladder_with_token_salad(self) -> None:
        printable_ascii_ladder = "".join(chr(codepoint) for codepoint in range(33, 127))
        output = (
            printable_ascii_ladder
            + "\n����<|vision_start|><|paratext|><|quad_start|>"
            + " async await constexpr namespace tokenizer logits processor vocabulary"
            + " ሜꌼﲇ𝓢햬𐍄㉿ﭹ"
        )

        self.assertIsNotNone(miner.gibberish_reason(output))

    def test_rejects_exact_native_smoke_artifact(self) -> None:
        output = _smoke_gibberish()

        self.assertIsNotNone(miner.gibberish_reason(output))

    def test_rejects_ascending_completion_token_ids(self) -> None:
        choice = SimpleNamespace(token_ids=list(range(96)))

        self.assertEqual(
            miner.gibberish_reason("apparently plausible text", choice),
            "ascending MinerU vocabulary walk",
        )

    def test_rejects_rare_unicode_tail_completion_token_ids(self) -> None:
        token_ids = [150_400 + (index * 37) % 1_243 for index in range(96)]
        choice = SimpleNamespace(token_ids=token_ids)

        self.assertEqual(
            miner.gibberish_reason("���" + "漢" * 140, choice),
            "rare-Unicode MinerU vocabulary soup",
        )

    def test_accepts_plausible_ocr_across_supported_content(self) -> None:
        cases = {
            "english": (
                "ARTICLE 12 — WAGES\nEmployees shall receive the hourly rates "
                "listed below. The Employer and Union agree that these rates "
                "remain effective through June 30, 2028."
            ),
            "markdown_table": (
                "| Classification | Step 1 | Step 2 |\n"
                "| --- | ---: | ---: |\n"
                "| Technician | $24.50 | $25.75 |\n"
                "| Operator | $22.10 | $23.40 |"
            ),
            "math": (
                "Overtime premium: $R_o = 1.5 \\times R_b$. "
                "For hours $h > 40$, total pay is $40R_b + (h-40)R_o$."
            ),
            "cjk": (
                "第十二条 工资与工作时间\n雇员每周工作四十小时，超过规定时间的工作按加班计算。"
                "労働者と使用者は、賃金表を毎年見直すことに合意する。"
            ),
            "mixed_language": (
                "Convention collective / Collective Agreement\n"
                "Les employés — employees — recibirán un ajuste salarial del 3 %. "
                "工资表 Wage Schedule 2027–2028."
            ),
        }

        for name, output in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(miner.gibberish_reason(output))


class MinerGibberishIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_finished_gibberish_retries_then_raises_quality_error(self) -> None:
        completions = AlwaysSameCompletions(_smoke_gibberish(), finish_reason="stop")
        args = miner.parse_args([])
        context = _make_context(args, completions)

        with self.assertRaises(GenerationQualityError) as caught:
            await common.request_chat_completion(
                context,
                {
                    "model": args.model_name,
                    "messages": [],
                    "temperature": 0,
                    "presence_penalty": 1.0,
                    "frequency_penalty": 0.05,
                },
                timeout=600,
            )

        self.assertEqual(len(completions.calls), 3)
        self.assertIn("source/doc page 1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
