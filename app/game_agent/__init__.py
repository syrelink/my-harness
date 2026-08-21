"""GameRover Agent 包的公共入口。

这是一个带上下文预算控制的单 Agent 游戏信息 Harness（无 LangGraph）。
外部模块只需要从这里导入构建函数。
"""

from app.game_agent.agent import GameAgent, build_game_assistant

__all__ = ["GameAgent", "build_game_assistant"]
