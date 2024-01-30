from collections import deque
from datetime import datetime
from enum import Enum

from dateutil.relativedelta import relativedelta


class Type(Enum):
    MaxLoss = 1
    MaxDrawDown = 2


class Period(Enum):
    Month = 30
    Quarter = 90
    YTD = 365


class Action(Enum):
    FlatPositionSuspendTrading = 1
    ReduceCapitalAndLimits = 2


class Scope(Enum):
    Symbol = 1
    Strategy = 2


class RiskRule:
    def __init__(self, type, lossAmount, period, action, reduceCapitalAndLimitsRatio, scope):
        self.type = type
        self.lossAmount = lossAmount
        self.period = period
        self.action = action
        self.reduceCapitalAndLimitsRatio = reduceCapitalAndLimitsRatio
        self.scope = scope
        self.triggered = False

    def __str__(self):
        return f"RiskRule(Type={self.type}, lossAmount={self.lossAmount}, period={self.period}, action={self.action}, reduceCapitalAndLimitsRatio={self.reduceCapitalAndLimitsRatio}, scope={self.scope})"

    def checkRule(self, acct_ccy_pnl, curr_date, datetime_index):
        # if no account currency pnl then return None
        if not acct_ccy_pnl:
            return None

        curr_value = acct_ccy_pnl[-1]

        if self.period == Period.YTD:
            period_start_date = datetime(curr_date.year, 1, 1)  # beginning of the year
        elif self.period == Period.Month:
            period_start_date = datetime(curr_date.year, curr_date.month, 1)  # beginning of the month
        elif self.period == Period.Quarter:
            quarter = (curr_date.month - 1) // 3 + 1
            period_start_date = datetime(curr_date.year, 3 * quarter - 2, 1)  # beginning of the quarter

        period_start_index = datetime_index.get_loc(period_start_date, method="nearest")
        period_start_value = acct_ccy_pnl[period_start_index]

        # if rule type is MaxLoss and loss is more than lossAmount then return action
        if self.type == Type.MaxLoss and (curr_value - period_start_value) < -self.lossAmount:
            return self.action

        # if rule type is MaxDrawDown
        if self.type == Type.MaxDrawDown:
            # get start values
            start_values = acct_ccy_pnl[period_start_index:]
            if len(start_values) > 0:
                max_drawdown = min([curr_value - start for start in start_values])
            else:
                max_drawdown = 0

            if max_drawdown < -self.lossAmount:
                return self.action

        return None


class TriggeredRule:
    def __init__(self, start_timestamp, end_timestamp, symbol, rule):
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.symbol = symbol
        self.rule = rule

    def __str__(self):
        return f"TriggeredRule(start_timestamp={self.start_timestamp}, end_timestamp={self.end_timestamp}, symbol={self.symbol}, rule={str(self.rule)})"


class RiskRuleProcessor:
    class SymbolDetails:
        def __init__(self, tradingSuspended, betSize, instrumentGrossLimit):
            self.TradingSuspended = tradingSuspended
            self.BetSize = betSize
            self.InstrumentGrossLimit = instrumentGrossLimit

        def __str__(self):
            return f"SymbolDetails(TradingSuspended={self.TradingSuspended}, BetSize={self.BetSize})"

    def __init__(self, betSize, grossLimit, instrumentGrossLimit, nopLimit):
        self.riskRules = []
        self.triggeredRules = deque()
        self.BetSize = betSize
        self.GrossLimit = grossLimit
        self.InstrumentGrossLimit = instrumentGrossLimit
        self.NOPLimit = nopLimit
        self.TradingSuspended = False
        self.symbolDetails = {}

    def __str__(self):
        return f"RiskRuleProcessor(BetSize={self.BetSize}, GrossLimit={self.GrossLimit}, InstrumentGrossLimit={self.InstrumentGrossLimit}, TradingSuspended={self.TradingSuspended}, symbolDetails={self.symbolDetails}, riskRules=[{', '.join(map(str, self.riskRules))}], triggeredRules=[{', '.join(map(str, self.triggeredRules))}])"

    def addRiskRule(self, type, lossAmount, period, action, reduceCapitalAndLimitsRatio, scope):
        self.riskRules.append(RiskRule(type, lossAmount, period, action, reduceCapitalAndLimitsRatio, scope))

    def addRiskRules(self, risk_rules):
        self.riskRules.extend(risk_rules)

    def applyRiskRules(self, trading_stats_by_symbol, strategy_stats, checkDateTime, date_to_index, callback=None):
        for symbol, stats in trading_stats_by_symbol.items():
            if symbol in self.symbolDetails and self.symbolDetails[symbol].TradingSuspended:
                continue
            for rule in self.riskRules:
                if rule.scope == Scope.Symbol and not rule.triggered:
                    action = rule.checkRule(stats["acct_ccy_pnl"], checkDateTime, date_to_index)
                    if action:
                        self.applyAction(symbol, rule, action.value)
                        period_end_time = self.calculatePeriodEndTime(rule, checkDateTime)
                        self.triggeredRules.append(TriggeredRule(checkDateTime, period_end_time, symbol, rule))
                        if callback:
                            callback(symbol, rule)

        if not self.TradingSuspended:
            for rule in self.riskRules:
                if rule.scope == Scope.Strategy and not rule.triggered:
                    action = rule.checkRule(strategy_stats["acct_ccy_pnl"], checkDateTime, date_to_index)
                    if action:
                        self.applyAction(None, rule, action.value)
                        period_end_time = self.calculatePeriodEndTime(rule, checkDateTime)
                        self.triggeredRules.append(TriggeredRule(checkDateTime, period_end_time, None, rule))
                        if callback:
                            callback(None, rule)

        self.cleanTriggeredRules(checkDateTime)

    def cleanTriggeredRules(self, checkDateTime):
        for rule in list(self.triggeredRules):
            if checkDateTime > rule.end_timestamp:
                self.undoAction(rule.symbol, rule.rule)
                self.triggeredRules.remove(rule)

    def applyAction(self, symbol, rule, action):
        rule.triggered = True
        if symbol:
            symbol_details = self.getSymbolDetails(symbol)
            if rule.scope == Scope.Symbol:
                if action == 1:
                    symbol_details.TradingSuspended = True
                elif action == 2:
                    symbol_details.BetSize = self.BetSize * rule.reduceCapitalAndLimitsRatio
                    symbol_details.InstrumentGrossLimit *= rule.reduceCapitalAndLimitsRatio

        elif rule.scope == Scope.Strategy:
            if action == 1:
                self.TradingSuspended = True
            elif action == 2:
                self.GrossLimit *= rule.reduceCapitalAndLimitsRatio
                self.InstrumentGrossLimit *= rule.reduceCapitalAndLimitsRatio
                self.NOPLimit *= rule.reduceCapitalAndLimitsRatio
                self.BetSize *= rule.reduceCapitalAndLimitsRatio

    def undoAction(self, symbol, rule):
        rule.triggered = False
        if symbol:
            symbol_details = self.getSymbolDetails(symbol)
            if rule.scope == Scope.Symbol:
                if rule.action.value == 1:
                    symbol_details.TradingSuspended = False
                elif rule.action.value == 2:
                    symbol_details.BetSize = self.BetSize
                    symbol_details.InstrumentGrossLimit = self.InstrumentGrossLimit
        elif rule.scope == Scope.Strategy:
            if rule.action.value == 1:
                self.TradingSuspended = False
            elif rule.action.value == 2:
                self.GrossLimit /= rule.reduceCapitalAndLimitsRatio
                self.InstrumentGrossLimit /= rule.reduceCapitalAndLimitsRatio
                self.NOPLimit /= rule.reduceCapitalAndLimitsRatio
                self.BetSize /= rule.reduceCapitalAndLimitsRatio

    def getSymbolDetails(self, symbol):
        if symbol not in self.symbolDetails:
            self.symbolDetails[symbol] = self.SymbolDetails(False, self.BetSize, self.InstrumentGrossLimit)
        return self.symbolDetails[symbol]

    def isTradingSuspended(self, symbol=None):
        if self.TradingSuspended:
            return True
        elif symbol and symbol in self.symbolDetails:
            return self.symbolDetails[symbol].TradingSuspended
        else:
            return False

    def calculatePeriodEndTime(self, rule, checkDateTime):
        if rule.period == Period.Month:
            period_end_time = datetime(checkDateTime.year, checkDateTime.month, 1) + relativedelta(months=1, days=-1)
        elif rule.period == Period.Quarter:
            quarter = (checkDateTime.month - 1) // 3 + 1
            month = 3 * quarter
            period_end_time = datetime(checkDateTime.year, month, 1) + relativedelta(months=1, days=-1)
        elif rule.period == Period.YTD:
            period_end_time = datetime(checkDateTime.year, 12, 31)
        return period_end_time
