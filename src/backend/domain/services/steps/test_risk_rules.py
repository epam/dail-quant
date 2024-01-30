import csv
import io
import unittest
from collections import defaultdict
from datetime import datetime, timedelta

from risk_limit import *

# date,     stock1PnL,stock2PnL,stratPnL,stock1State,stock2State,stratState,stock1BetSize,stock2BetSize,expected_gross_limit
csv_data = """
2022-01-01,500,600,1100,active,active,active,100,100,10000
2022-01-02,500,600,1100,active,active,active,100,100,10000
2022-01-03,505,605,110,active,active,suspended,100,100,10000
2022-01-08,505,605,110,active,active,suspended,100,100,10000
2022-03-03,505,605,110,active,active,active,100,100,10000
"""


class TestRiskRuleProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = RiskRuleProcessor(100, 10000, 5000, 20000)
        self.processor.addRiskRule("MaxLoss", 500, "Month", "FlatPositionSuspendTrading", 0.5, "Symbol")
        self.processor.addRiskRule("MaxDrawDown", 500, "Month", "FlatPositionSuspendTrading", 0.5, "Strategy")

    def test_applyRiskRules_with_csvstring(self):
        trading_stats_by_symbol = defaultdict(lambda: defaultdict(list))
        strategy_stats = defaultdict(list)
        previous_day_data = None

        reader = csv.reader(io.StringIO(csv_data.strip()))
        for row in reader:
            (
                date_str,
                stock1_pnl,
                stock2_pnl,
                strategy_pnl,
                stock1_state,
                stock2_state,
                strategy_state,
                stock1_bet_size,
                stock2_bet_size,
                expected_gross_limit,
            ) = row

            date = datetime.strptime(date_str, "%Y-%m-%d").date()

            if previous_day_data:
                previous_date = previous_day_data[0]
                while previous_date + timedelta(days=1) < date:
                    previous_date += timedelta(days=1)

                    trading_stats_by_symbol["stock1"]["acc_curr_pnl"].append(previous_day_data[1])
                    trading_stats_by_symbol["stock2"]["acc_curr_pnl"].append(previous_day_data[2])
                    strategy_stats["acc_curr_pnl"].append(previous_day_data[3])

                    self.processor.applyRiskRules(trading_stats_by_symbol, strategy_stats, previous_date)

            trading_stats_by_symbol["stock1"]["acc_curr_pnl"].append(int(stock1_pnl))
            trading_stats_by_symbol["stock2"]["acc_curr_pnl"].append(int(stock2_pnl))
            strategy_stats["acc_curr_pnl"].append(int(strategy_pnl))

            self.processor.applyRiskRules(trading_stats_by_symbol, strategy_stats, date)

            self.assertEqual(self.processor.getSymbolDetails("stock1").TradingSuspended, stock1_state == "suspended")
            self.assertEqual(self.processor.getSymbolDetails("stock2").TradingSuspended, stock2_state == "suspended")
            self.assertEqual(self.processor.TradingSuspended, strategy_state == "suspended")
            self.assertEqual(self.processor.getSymbolDetails("stock1").BetSize, float(stock1_bet_size))
            self.assertEqual(self.processor.getSymbolDetails("stock2").BetSize, float(stock2_bet_size))
            self.assertEqual(self.processor.GrossLimit, float(expected_gross_limit))

            previous_day_data = (date, int(stock1_pnl), int(stock2_pnl), int(strategy_pnl))


if __name__ == "__main__":
    unittest.main()
