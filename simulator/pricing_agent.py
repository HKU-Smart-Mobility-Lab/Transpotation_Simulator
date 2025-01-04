# Andrew, 2024/12/29
class PricingAgent:
    def __init__(self, strategy="static", learning_rate=0.01):
        """
        初始化PricingAgent
        :param strategy: 定价策略，"dynamic"、"static"
        :param learning_rate: 学习率
        """
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.price_table = {}  # 保存不同区域的价格信息（可以扩展为更复杂的结构）
        
    def get_action(self, pricing_state):
        """
        根据当前状态生成价格决策
        :param pricing_state: dict 包含供需信息、历史价格等
        :return: float 动态生成的价格
        """
        if self.strategy == "dynamic":
            # 示例动态定价逻辑：根据供需比动态调整价格
            supply = pricing_state.get("supply", 1)
            demand = pricing_state.get("demand", 1)
            base_price = pricing_state.get("base_price", 2.5)
            return base_price * (1 + 0.1 * (demand - supply) / max(supply, demand))
        elif self.strategy == "static":
            # 静态定价逻辑，基于提供的定价公式
            trip_distance = pricing_state.get("trip_distance", 0)  # 单位：公里
            base_price = 2.5
            additional_price = 0.5 * int(max(0, trip_distance * 1000 - 322) / 322)
            return base_price + additional_price
        else:
            raise ValueError("Unsupported pricing strategy")

    def update(self, pricing_state, action, reward):
        """
        根据反馈调整策略
        :param pricing_state: dict 当前状态
        :param action: float 当前定价决策
        :param reward: float 来自市场的反馈（例如乘客的接受率）
        """
        # 示例更新逻辑：调整学习率并更新价格表
        location = pricing_state.get("location", "global")
        if location not in self.price_table:
            self.price_table[location] = action
        self.price_table[location] += self.learning_rate * (reward - self.price_table[location])

    @staticmethod
    def observe(simulator):
        """
        从Simulator中提取当前状态
        :param simulator: Simulator实例
        :return: dict 包含供需信息的状态
        """
        state = {
            "supply": simulator.get_driver_count(),
            "demand": simulator.get_request_count(),
            "base_price": 2.5,  # 可以根据情况动态获取
            "location": "global",  # 示例，全局定价
        }
        return state
