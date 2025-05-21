import logging
import sys
import sqlite3
from typing import Optional

from shared_types.models import TradingSignalModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

class JournalAgent:
    def __init__(self, db_path: Optional[str] = "trading_journal.sqlite"):
        logger.info(f"JournalAgent initializing with DB path: {db_path}")
        """
        Initializes the JournalAgent.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._ensure_db_and_tables_exist()

    def _ensure_db_and_tables_exist(self):
        """Ensures the SQLite database and necessary tables exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create signals table (adjust fields based on TradingSignalModel)
            # This is a simplified schema. Consider data types, constraints, and indexing.
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                signal_type TEXT NOT NULL, -- 'BUY' or 'SELL'
                price REAL NOT NULL,
                source TEXT, -- e.g., 'BreakoutStrategy'
                confidence REAL, -- Optional
                stop_loss REAL, -- Optional
                take_profit REAL, -- Optional
                risk_reward_ratio REAL, -- Optional
                position_size_units REAL, -- Optional
                position_size_usd REAL, -- Optional
                risk_per_trade_usd REAL, -- Optional
                status TEXT DEFAULT 'PENDING' -- e.g., PENDING, APPROVED, REJECTED, EXECUTED, CLOSED
                -- Add other fields from TradingSignalModel as needed
            )
            """)
            conn.commit()
            logger.info(f"Database '{self.db_path}' and table 'signals' ensured.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during table creation in '{self.db_path}': {e}")
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    def record_signal(self, signal: TradingSignalModel) -> bool:
        """Records an approved trading signal to the journal."""
        logger.info(f"Attempting to record signal {signal.signal_id} for {signal.asset.value} {signal.signal_type.value} @ {signal.entry_price}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the signal already exists to avoid duplicates
            cursor.execute("SELECT 1 FROM signals WHERE signal_id = ?", (signal.signal_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                logger.info(f"Signal {signal.signal_id} already exists in the journal, skipping record")
                return True

            # Assuming 'APPROVED' status when RiskManagerAgent calls this
            signal_status = "APPROVED"

            sql = """
            INSERT INTO signals (
                signal_id, timestamp, asset, signal_type, price,
                source, confidence, stop_loss, take_profit,
                risk_reward_ratio, position_size_units, position_size_usd,
                risk_per_trade_usd, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                signal.signal_id,
                signal.generated_at.isoformat(),
                signal.asset.value,
                signal.signal_type.value,
                signal.entry_price,
                signal.metadata.get('strategy_name', 'UnknownStrategy'), # Example for source
                signal.metadata.get('confidence'), # Optional
                signal.stop_loss,
                signal.take_profit,
                signal.risk_reward_ratio,
                signal.position_size_asset, # Mapped to position_size_units
                signal.position_size_usd,
                signal.risk_per_trade_usd,
                signal_status
            )
            
            cursor.execute(sql, params)
            conn.commit()
            logger.info(f"Signal {signal.signal_id} recorded successfully to database.")
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite error while recording signal {signal.signal_id}: {e}", exc_info=True)
            return False
        except AttributeError as e:
            logger.error(f"Attribute error, likely missing field in signal object for {signal.signal_id}: {e}", exc_info=True)
            return False
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    def record_trade(self, trade_data: dict): 
        """
        Records a trade execution into the journal.
        Placeholder for now.
        """
        logger.info(f"Recording trade: {trade_data} (placeholder).")
        # Implementation to write to a 'trades' table

    # Add other methods for querying, statistics as per Memory a26e5920-cc65-4a95-97f8-5d697a3956c4
    # E.g., get_signal_by_id, get_all_signals, get_trade_statistics, etc.
