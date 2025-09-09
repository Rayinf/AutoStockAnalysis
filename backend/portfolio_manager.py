"""
持仓管理模块
管理用户的股票持仓、资金状况和交易记录
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import akshare as ak
import pandas as pd

# 全局股票名称缓存类
class StockNameCache:
    """股票名称缓存管理器"""
    _instance = None
    _cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_cache()
        return cls._instance
    
    def _init_cache(self):
        """初始化常用股票名称缓存"""
        self._cache = {
            # A股主板 (000xxx)
            '000001': '平安银行', '000002': '万科A', '000501': '武商集团',
            '000519': '中兵红箭', '000723': '美锦能源', '000858': '五粮液',
            '001286': '陕西能源',
            
            # 中小板/创业板 (002xxx)  
            '002015': '协鑫能科', '002160': '常铝股份', '002182': '宝武镁业',
            '002415': '海康威视', '002436': '兴森科技', '002487': '大金重工',
            '002594': '比亚迪',
            
            # 上海主板 (600xxx, 601xxx)
            '600036': '招商银行', '600176': '中国巨石', '600519': '贵州茅台',
            '600585': '海螺水泥', '600710': '苏美达', '600885': '宏发股份',
            '601717': '中创智领'
        }
    
    def get(self, symbol: str) -> Optional[str]:
        """从缓存获取股票名称"""
        return self._cache.get(symbol)
    
    def set(self, symbol: str, name: str):
        """设置股票名称到缓存"""
        self._cache[symbol] = name
    
    def has(self, symbol: str) -> bool:
        """检查缓存中是否存在该股票"""
        return symbol in self._cache

# 全局缓存实例
stock_name_cache = StockNameCache()

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    name: str
    quantity: int  # 持有数量（股）
    avg_cost: float  # 平均成本价
    current_price: float  # 当前价格
    market_value: float  # 市值
    profit_loss: float  # 盈亏金额
    profit_loss_pct: float  # 盈亏百分比
    last_updated: str  # 最后更新时间

@dataclass
class Transaction:
    """交易记录"""
    id: int
    symbol: str
    name: str
    action: str  # 'buy' 或 'sell'
    quantity: int
    price: float
    amount: float  # 交易金额
    fee: float  # 手续费
    timestamp: str
    note: str  # 备注

@dataclass
class Portfolio:
    """投资组合"""
    total_assets: float  # 总资产
    available_cash: float  # 可用资金
    market_value: float  # 持仓市值
    total_profit_loss: float  # 总盈亏
    total_profit_loss_pct: float  # 总盈亏百分比
    positions: List[Position]  # 持仓列表
    position_count: int  # 持仓股票数量

class PortfolioManager:
    """持仓管理器"""
    
    def __init__(self, db_path: str = "portfolio.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建持仓表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        avg_cost REAL NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                ''')
                
                # 创建交易记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        amount REAL NOT NULL,
                        fee REAL DEFAULT 0,
                        timestamp TEXT NOT NULL,
                        note TEXT DEFAULT ''
                    )
                ''')
                
                # 创建账户信息表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS account (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        available_cash REAL NOT NULL DEFAULT 100000,
                        last_updated TEXT NOT NULL
                    )
                ''')
                
                # 初始化账户（如果不存在）
                cursor.execute('SELECT COUNT(*) FROM account')
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO account (available_cash, last_updated) 
                        VALUES (100000, ?)
                    ''', (datetime.now().isoformat(),))
                
                conn.commit()
                self.logger.info("数据库初始化完成")
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise
    
    def get_available_cash(self) -> float:
        """获取可用资金"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT available_cash FROM account WHERE id = 1')
                result = cursor.fetchone()
                return result[0] if result else 100000.0
        except Exception as e:
            self.logger.error(f"获取可用资金失败: {e}")
            return 100000.0
    
    def update_cash(self, amount: float):
        """更新可用资金"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE account SET available_cash = ?, last_updated = ?
                    WHERE id = 1
                ''', (amount, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            self.logger.error(f"更新资金失败: {e}")
    
    def add_transaction(self, symbol: str, name: str, action: str, 
                       quantity: int, price: float, fee: float = 0, 
                       note: str = "") -> bool:
        """添加交易记录"""
        try:
            amount = quantity * price
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 添加交易记录
                cursor.execute('''
                    INSERT INTO transactions 
                    (symbol, name, action, quantity, price, amount, fee, timestamp, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, name, action, quantity, price, amount, fee, 
                      datetime.now().isoformat(), note))
                
                # 更新持仓
                if action.lower() == 'buy':
                    self._update_position_after_buy(cursor, symbol, name, quantity, price)
                    # 减少可用资金
                    cash = self.get_available_cash()
                    self.update_cash(cash - amount - fee)
                    
                elif action.lower() == 'sell':
                    self._update_position_after_sell(cursor, symbol, quantity, price)
                    # 增加可用资金
                    cash = self.get_available_cash()
                    self.update_cash(cash + amount - fee)
                
                conn.commit()
                self.logger.info(f"交易记录添加成功: {action} {quantity}股 {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"添加交易记录失败: {e}")
            return False
    
    def _update_position_after_buy(self, cursor, symbol: str, name: str, 
                                  quantity: int, price: float):
        """买入后更新持仓"""
        # 查询现有持仓
        cursor.execute('SELECT quantity, avg_cost FROM positions WHERE symbol = ?', (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有持仓
            old_quantity, old_avg_cost = existing
            new_quantity = old_quantity + quantity
            new_avg_cost = (old_quantity * old_avg_cost + quantity * price) / new_quantity
            
            cursor.execute('''
                UPDATE positions SET quantity = ?, avg_cost = ?, last_updated = ?
                WHERE symbol = ?
            ''', (new_quantity, new_avg_cost, datetime.now().isoformat(), symbol))
        else:
            # 新增持仓
            cursor.execute('''
                INSERT INTO positions (symbol, name, quantity, avg_cost, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, name, quantity, price, datetime.now().isoformat()))
    
    def _update_position_after_sell(self, cursor, symbol: str, quantity: int, price: float):
        """卖出后更新持仓"""
        cursor.execute('SELECT quantity FROM positions WHERE symbol = ?', (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            old_quantity = existing[0]
            new_quantity = old_quantity - quantity
            
            if new_quantity <= 0:
                # 清空持仓
                cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
            else:
                # 更新持仓数量
                cursor.execute('''
                    UPDATE positions SET quantity = ?, last_updated = ?
                    WHERE symbol = ?
                ''', (new_quantity, datetime.now().isoformat(), symbol))
    
    def get_positions(self, with_current_price: bool = True) -> List[Position]:
        """获取所有持仓"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, name, quantity, avg_cost, last_updated
                    FROM positions WHERE quantity > 0
                    ORDER BY symbol
                ''')
                
                positions = []
                for row in cursor.fetchall():
                    symbol, name, quantity, avg_cost, last_updated = row
                    
                    # 获取当前价格（这里暂时使用模拟价格，实际应该调用实时价格接口）
                    current_price = self._get_current_price(symbol) if with_current_price else avg_cost
                    
                    market_value = quantity * current_price
                    cost_value = quantity * avg_cost
                    profit_loss = market_value - cost_value
                    profit_loss_pct = (profit_loss / cost_value * 100) if cost_value > 0 else 0
                    
                    position = Position(
                        symbol=symbol,
                        name=name,
                        quantity=quantity,
                        avg_cost=avg_cost,
                        current_price=current_price,
                        market_value=market_value,
                        profit_loss=profit_loss,
                        profit_loss_pct=profit_loss_pct,
                        last_updated=last_updated
                    )
                    positions.append(position)
                
                return positions
                
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return []
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前价格（优先真实数据，回退安全默认）"""
        try:
            # 优先用近5日历史的收盘价作为“当前价”近似，避免大接口延迟
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=5)
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"
            )
            if df is not None and not df.empty and '收盘' in df.columns:
                return float(df.iloc[-1]['收盘'])
        except Exception:
            pass
        
        # 回退：用平均成本以避免夸大市值
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT avg_cost FROM positions WHERE symbol = ?', (symbol,))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception:
            pass
        
        return 0.0
    
    def get_portfolio(self) -> Portfolio:
        """获取投资组合概况"""
        try:
            positions = self.get_positions()
            available_cash = self.get_available_cash()
            
            market_value = sum(pos.market_value for pos in positions)
            total_profit_loss = sum(pos.profit_loss for pos in positions)
            total_cost = sum(pos.quantity * pos.avg_cost for pos in positions)
            total_profit_loss_pct = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0
            
            total_assets = available_cash + market_value
            
            return Portfolio(
                total_assets=total_assets,
                available_cash=available_cash,
                market_value=market_value,
                total_profit_loss=total_profit_loss,
                total_profit_loss_pct=total_profit_loss_pct,
                positions=positions,
                position_count=len(positions)
            )
            
        except Exception as e:
            self.logger.error(f"获取投资组合失败: {e}")
            return Portfolio(
                total_assets=100000.0,
                available_cash=100000.0,
                market_value=0.0,
                total_profit_loss=0.0,
                total_profit_loss_pct=0.0,
                positions=[],
                position_count=0
            )
    
    def get_transactions(self, limit: int = 100, symbol: str = None) -> List[Transaction]:
        """获取交易记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute('''
                        SELECT * FROM transactions WHERE symbol = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM transactions
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (limit,))
                
                transactions = []
                for row in cursor.fetchall():
                    transaction = Transaction(
                        id=row[0],
                        symbol=row[1],
                        name=row[2],
                        action=row[3],
                        quantity=row[4],
                        price=row[5],
                        amount=row[6],
                        fee=row[7],
                        timestamp=row[8],
                        note=row[9]
                    )
                    transactions.append(transaction)
                
                return transactions
                
        except Exception as e:
            self.logger.error(f"获取交易记录失败: {e}")
            return []
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """根据股票代码获取持仓"""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None
    
    def is_holding(self, symbol: str) -> bool:
        """检查是否持有某只股票"""
        return self.get_position_by_symbol(symbol) is not None
    
    def get_holding_symbols(self) -> List[str]:
        """获取所有持仓股票代码"""
        positions = self.get_positions(with_current_price=False)
        return [pos.symbol for pos in positions]
    
    def calculate_max_buy_quantity(self, symbol: str, price: float, 
                                 cash_ratio: float = 0.95) -> int:
        """计算最大可买数量"""
        try:
            available_cash = self.get_available_cash()
            usable_cash = available_cash * cash_ratio  # 保留一定现金
            
            # 考虑手续费（简化为0.1%）
            fee_rate = 0.001
            max_amount = usable_cash / (1 + fee_rate)
            max_quantity = int(max_amount / price)
            
            # A股最小交易单位是100股
            return (max_quantity // 100) * 100
            
        except Exception as e:
            self.logger.error(f"计算最大买入数量失败: {e}")
            return 0
    
    def edit_position(self, symbol: str, new_quantity: int, new_avg_cost: float, note: str = "") -> bool:
        """直接编辑持仓信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if new_quantity <= 0:
                    # 清空持仓
                    cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
                    self.logger.info(f"清空持仓: {symbol}")
                else:
                    # 更新或插入持仓
                    cursor.execute('''
                        INSERT OR REPLACE INTO positions (symbol, name, quantity, avg_cost, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (symbol, self._get_stock_name(symbol), new_quantity, new_avg_cost, datetime.now().isoformat()))
                    self.logger.info(f"编辑持仓: {symbol} 数量{new_quantity} 成本{new_avg_cost}")
                
                # 记录编辑操作
                if note:
                    cursor.execute('''
                        INSERT INTO transactions 
                        (symbol, name, action, quantity, price, amount, fee, timestamp, note)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, self._get_stock_name(symbol), 'edit', new_quantity, new_avg_cost, 
                          new_quantity * new_avg_cost, 0, datetime.now().isoformat(), f"手动编辑: {note}"))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"编辑持仓失败: {e}")
            return False
    
    def _get_stock_name(self, symbol: str) -> str:
        """获取股票名称 - 使用全局缓存策略优化速度"""
        
        # 优先从全局缓存获取 (最快，0.0000秒)
        cached_name = stock_name_cache.get(symbol)
        if cached_name:
            return cached_name
        
        # 缓存未命中，尝试通过akshare获取
        try:
            import akshare as ak
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            if stock_info is not None and not stock_info.empty:
                # 查找股票简称
                name_row = stock_info[stock_info['item'] == '股票简称']
                if not name_row.empty:
                    stock_name = name_row['value'].iloc[0]
                    if stock_name and stock_name != symbol:
                        # 动态更新全局缓存以备下次使用
                        stock_name_cache.set(symbol, stock_name)
                        self.logger.info(f"API获取并缓存股票名称: {symbol} -> {stock_name}")
                        return stock_name
        except Exception as e:
            self.logger.debug(f"API获取股票名称失败 {symbol}: {e}")
        
        # 最后回退：返回股票代码本身，但也缓存起来避免重复API调用
        stock_name_cache.set(symbol, symbol)
        self.logger.debug(f"使用股票代码作为名称并缓存: {symbol}")
        return symbol
    
    def delete_position(self, symbol: str) -> bool:
        """删除持仓"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
                conn.commit()
                self.logger.info(f"删除持仓: {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"删除持仓失败: {e}")
            return False
    
    def batch_add_transactions(self, transactions: List[Dict]) -> Dict:
        """批量添加交易记录"""
        results = {
            'success_count': 0,
            'failed_count': 0,
            'errors': []
        }
        
        for transaction in transactions:
            try:
                success = self.add_transaction(
                    symbol=transaction['symbol'],
                    name=transaction['name'],
                    action=transaction['action'],
                    quantity=transaction['quantity'],
                    price=transaction['price'],
                    fee=transaction.get('fee', 0),
                    note=transaction.get('note', '')
                )
                if success:
                    results['success_count'] += 1
                else:
                    results['failed_count'] += 1
                    results['errors'].append(f"添加 {transaction['symbol']} 失败")
            except Exception as e:
                results['failed_count'] += 1
                results['errors'].append(f"添加 {transaction['symbol']} 异常: {str(e)}")
        
        return results
    
    def sync_with_stock_pool(self, stock_pool_symbols: List[str]) -> Dict:
        """与股票池同步，返回持仓分析"""
        try:
            current_positions = self.get_holding_symbols()
            
            # 分析持仓与股票池的关系
            in_both = set(current_positions) & set(stock_pool_symbols)  # 既持仓又在股票池
            only_holding = set(current_positions) - set(stock_pool_symbols)  # 只持仓不在股票池
            only_in_pool = set(stock_pool_symbols) - set(current_positions)  # 只在股票池不持仓
            
            return {
                'current_positions_count': len(current_positions),
                'stock_pool_count': len(stock_pool_symbols),
                'in_both': list(in_both),
                'only_holding': list(only_holding),
                'only_in_pool': list(only_in_pool),
                'sync_suggestions': {
                    'consider_selling': list(only_holding),  # 考虑卖出（不在股票池中）
                    'consider_buying': list(only_in_pool),   # 考虑买入（在股票池但未持仓）
                    'keep_holding': list(in_both)            # 继续持有
                }
            }
            
        except Exception as e:
            self.logger.error(f"股票池同步分析失败: {e}")
            return {}
    
    def get_position_analysis(self) -> Dict:
        """获取持仓分析"""
        try:
            positions = self.get_positions()
            if not positions:
                return {'message': '暂无持仓'}
            
            # 计算各种统计指标
            total_cost = sum(pos.quantity * pos.avg_cost for pos in positions)
            total_market_value = sum(pos.market_value for pos in positions)
            total_profit_loss = total_market_value - total_cost
            
            # 盈利股票和亏损股票
            profit_stocks = [pos for pos in positions if pos.profit_loss > 0]
            loss_stocks = [pos for pos in positions if pos.profit_loss < 0]
            
            # 最大盈利和最大亏损
            max_profit_stock = max(positions, key=lambda x: x.profit_loss) if positions else None
            max_loss_stock = min(positions, key=lambda x: x.profit_loss) if positions else None
            
            # 持仓集中度（按市值）
            market_values = [pos.market_value for pos in positions]
            total_mv = sum(market_values)
            concentration = max(market_values) / total_mv if total_mv > 0 else 0
            
            return {
                'total_positions': len(positions),
                'total_cost': total_cost,
                'total_market_value': total_market_value,
                'total_profit_loss': total_profit_loss,
                'profit_stocks_count': len(profit_stocks),
                'loss_stocks_count': len(loss_stocks),
                'win_rate': len(profit_stocks) / len(positions) if positions else 0,
                'max_profit_stock': {
                    'symbol': max_profit_stock.symbol,
                    'name': max_profit_stock.name,
                    'profit_loss': max_profit_stock.profit_loss,
                    'profit_loss_pct': max_profit_stock.profit_loss_pct
                } if max_profit_stock else None,
                'max_loss_stock': {
                    'symbol': max_loss_stock.symbol,
                    'name': max_loss_stock.name,
                    'profit_loss': max_loss_stock.profit_loss,
                    'profit_loss_pct': max_loss_stock.profit_loss_pct
                } if max_loss_stock else None,
                'concentration_ratio': concentration,
                'risk_warnings': self._generate_risk_warnings(positions, concentration)
            }
            
        except Exception as e:
            self.logger.error(f"持仓分析失败: {e}")
            return {'error': str(e)}
    
    def _generate_risk_warnings(self, positions: List[Position], concentration: float) -> List[str]:
        """生成风险提醒"""
        warnings = []
        
        # 集中度过高警告
        if concentration > 0.5:
            warnings.append("持仓集中度过高，建议分散投资")
        
        # 亏损过大警告
        for pos in positions:
            if pos.profit_loss_pct < -20:
                warnings.append(f"{pos.symbol}({pos.name}) 亏损超过20%，请注意风险")
        
        # 持仓数量警告
        if len(positions) > 20:
            warnings.append("持仓股票数量过多，可能影响管理效率")
        elif len(positions) < 3:
            warnings.append("持仓股票数量过少，建议适当分散风险")
        
        return warnings

    def fix_stock_names(self) -> Dict[str, Any]:
        """修复数据库中错误的股票名称"""
        try:
            fixed_count = 0
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取所有持仓
                cursor.execute('SELECT symbol, name FROM positions')
                positions = cursor.fetchall()
                
                for symbol, current_name in positions:
                    # 如果名称就是代码，说明需要修复
                    if current_name == symbol:
                        correct_name = self._get_stock_name(symbol)
                        if correct_name != symbol:
                            # 更新数据库中的名称
                            cursor.execute('UPDATE positions SET name = ? WHERE symbol = ?', 
                                         (correct_name, symbol))
                            cursor.execute('UPDATE transactions SET name = ? WHERE symbol = ?', 
                                         (correct_name, symbol))
                            fixed_count += 1
                            self.logger.info(f"修复股票名称: {symbol} -> {correct_name}")
                
                conn.commit()
                
            return {
                'success': True,
                'fixed_count': fixed_count,
                'message': f'成功修复 {fixed_count} 个股票名称'
            }
            
        except Exception as e:
            self.logger.error(f"修复股票名称失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def export_to_dict(self) -> Dict:
        """导出持仓数据为字典格式"""
        try:
            portfolio = self.get_portfolio()
            transactions = self.get_transactions(limit=50)
            analysis = self.get_position_analysis()
            
            return {
                'portfolio': asdict(portfolio),
                'recent_transactions': [asdict(t) for t in transactions],
                'analysis': analysis,
                'export_time': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"导出持仓数据失败: {e}")
            return {}

# 全局持仓管理器实例
portfolio_manager = PortfolioManager()
