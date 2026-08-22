"""Command-line runner for the four integration scenarios."""

import asyncio

from .pipeline_demo import FullDuplexIntegrationDemo


async def run_all_scenarios():
    """Run all scenarios in one initialized demo composition."""
    demo = FullDuplexIntegrationDemo()
    await demo.start()
    try:
        return [
            await demo.run_backchannel_scenario(),
            await demo.run_interrupt_scenario(),
            await demo.run_translation_scenario(),
            await demo.run_task_resume_scenario(),
        ]
    finally:
        await demo.shutdown()


def main() -> None:
    results = asyncio.run(run_all_scenarios())
    for result in results:
        print(f"{result.name}: {'PASS' if result.passed else 'FAIL'} {result.details}")
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
