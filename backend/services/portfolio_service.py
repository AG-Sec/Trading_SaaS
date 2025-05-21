from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from backend.models.portfolio import Portfolio, Position, Trade, AssetClass, TradeDirection, TradeStatus
from backend.models.user import User, SubscriptionTier
from backend.core.middleware import get_tier_limit
from backend.services.market_data_service import MarketDataService

# Configure logging
logger = logging.getLogger(__name__)

class PortfolioService:
    """Service for managing user portfolios, positions, and trades"""
    
    @staticmethod
    async def create_portfolio(db: Session, user: User, name: str, description: Optional[str] = None, starting_capital: float = 10000.0) -> Portfolio:
        """
        Create a new portfolio for a user.
        
        Args:
            db: Database session
            user: User instance
            name: Portfolio name
            description: Optional portfolio description
            starting_capital: Initial capital amount
            
        Returns:
            Newly created Portfolio instance
        """
        # Check if user has reached their portfolio limit
        user_portfolios = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
        max_portfolios = await get_tier_limit(user, "max_portfolios")
        
        if len(user_portfolios) >= max_portfolios:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You have reached your limit of {max_portfolios} portfolios. Upgrade your subscription to create more."
            )
        
        # Check for duplicate name
        existing_portfolio = db.query(Portfolio).filter(
            Portfolio.user_id == user.id,
            Portfolio.name == name
        ).first()
        
        if existing_portfolio:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A portfolio with this name already exists"
            )
        
        # Create new portfolio
        portfolio = Portfolio(
            user_id=user.id,
            name=name,
            description=description,
            starting_capital=starting_capital,
            current_value=starting_capital,
            cash_balance=starting_capital
        )
        
        try:
            db.add(portfolio)
            db.commit()
            db.refresh(portfolio)
            logger.info(f"Created portfolio '{name}' for user ID {user.id}")
            return portfolio
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating portfolio: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating portfolio: {str(e)}"
            )
    
    @staticmethod
    async def get_portfolio(db: Session, portfolio_id: int, user_id: int) -> Portfolio:
        """
        Get a specific portfolio by ID.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio to retrieve
            user_id: ID of the user making the request
            
        Returns:
            Portfolio instance
        """
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id
        ).first()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found or you do not have permission to view it"
            )
        
        return portfolio
    
    @staticmethod
    async def get_user_portfolios(db: Session, user_id: int) -> List[Portfolio]:
        """
        Get all portfolios for a user.
        
        Args:
            db: Database session
            user_id: ID of the user
            
        Returns:
            List of Portfolio instances
        """
        return db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
    
    @staticmethod
    async def update_portfolio(db: Session, portfolio_id: int, user_id: int, name: Optional[str] = None, description: Optional[str] = None) -> Portfolio:
        """
        Update portfolio details.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio to update
            user_id: ID of the user making the request
            name: New portfolio name
            description: New portfolio description
            
        Returns:
            Updated Portfolio instance
        """
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        # Update fields if provided
        if name is not None:
            # Check for duplicate name
            existing_portfolio = db.query(Portfolio).filter(
                Portfolio.user_id == user_id,
                Portfolio.name == name,
                Portfolio.id != portfolio_id
            ).first()
            
            if existing_portfolio:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A portfolio with this name already exists"
                )
            
            portfolio.name = name
            
        if description is not None:
            portfolio.description = description
        
        try:
            db.commit()
            db.refresh(portfolio)
            logger.info(f"Updated portfolio ID {portfolio_id} for user ID {user_id}")
            return portfolio
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating portfolio: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating portfolio: {str(e)}"
            )
    
    @staticmethod
    async def delete_portfolio(db: Session, portfolio_id: int, user_id: int) -> bool:
        """
        Delete a portfolio.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio to delete
            user_id: ID of the user making the request
            
        Returns:
            True if successful
        """
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        try:
            db.delete(portfolio)
            db.commit()
            logger.info(f"Deleted portfolio ID {portfolio_id} for user ID {user_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting portfolio: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting portfolio: {str(e)}"
            )
    
    @staticmethod
    async def create_position(
        db: Session, 
        portfolio_id: int, 
        user_id: int, 
        symbol: str, 
        asset_class: AssetClass,
        direction: TradeDirection,
        quantity: float,
        entry_price: float,
        strategy_id: Optional[int] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        max_risk_percent: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Position:
        """
        Create a new position in a portfolio by executing a trade.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio
            user_id: ID of the user making the request
            symbol: Asset symbol
            asset_class: Asset class
            direction: Trade direction
            quantity: Quantity to trade
            entry_price: Entry price
            strategy_id: Optional strategy ID
            stop_loss_price: Optional stop loss price
            take_profit_price: Optional take profit price
            max_risk_percent: Optional maximum risk percent
            notes: Optional trade notes
            tags: Optional tags
            
        Returns:
            Newly created Position instance
        """
        # Get portfolio
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        # Calculate costs
        cost_basis = quantity * entry_price
        fees = cost_basis * 0.001  # Simulate a 0.1% fee
        total_cost = cost_basis + fees
        
        # Check if sufficient funds
        if direction == TradeDirection.LONG and portfolio.cash_balance < total_cost:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient funds. Required: {total_cost}, Available: {portfolio.cash_balance}"
            )
        
        # Calculate position size percent of portfolio
        position_size_percent = (cost_basis / portfolio.current_value) * 100
        
        # Create position
        position = Position(
            portfolio_id=portfolio_id,
            symbol=symbol,
            asset_class=asset_class,
            direction=direction,
            quantity=quantity,
            average_entry_price=entry_price,
            current_price=entry_price,
            cost_basis=cost_basis,
            market_value=cost_basis,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            max_risk_percent=max_risk_percent,
            position_size_percent=position_size_percent,
            strategy_id=strategy_id
        )
        
        # Create trade
        trade = Trade(
            portfolio_id=portfolio_id,
            symbol=symbol,
            asset_class=asset_class,
            direction=direction,
            quantity=quantity,
            price=entry_price,
            status=TradeStatus.OPEN,
            fees=fees,
            total_cost=total_cost,
            strategy_id=strategy_id,
            notes=notes,
            tags=json.dumps(tags) if tags else None
        )
        
        try:
            # Add position and trade
            db.add(position)
            db.flush()  # Get position ID without committing
            
            # Link trade to position
            trade.position_id = position.id
            db.add(trade)
            
            # Update portfolio cash balance
            if direction == TradeDirection.LONG:
                portfolio.cash_balance -= total_cost
            else:
                # For short positions, assume margin requirements are met and cash is set aside
                portfolio.cash_balance -= (total_cost * 0.5)  # 50% margin requirement
            
            db.commit()
            db.refresh(position)
            logger.info(f"Created position in {symbol} for portfolio ID {portfolio_id}")
            
            return position
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating position: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating position: {str(e)}"
            )
    
    @staticmethod
    async def close_position(
        db: Session, 
        position_id: int, 
        user_id: int, 
        exit_price: float,
        quantity: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Trade:
        """
        Close a position (fully or partially) by executing a sell trade.
        
        Args:
            db: Database session
            position_id: ID of the position to close
            user_id: ID of the user making the request
            exit_price: Exit price
            quantity: Quantity to close (if None, close entire position)
            notes: Optional trade notes
            tags: Optional tags
            
        Returns:
            Trade instance representing the closing trade
        """
        # Get position
        position = db.query(Position).filter(Position.id == position_id).first()
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Position not found"
            )
        
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, position.portfolio_id, user_id)
        
        # Set quantity to full position if not specified
        if quantity is None or quantity >= position.quantity:
            quantity = position.quantity
            full_close = True
        else:
            full_close = False
            
        if quantity <= 0 or quantity > position.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quantity. Must be between 0 and {position.quantity}"
            )
        
        # Calculate trade details
        portion = quantity / position.quantity
        cost_basis_portion = position.cost_basis * portion
        market_value = quantity * exit_price
        fees = market_value * 0.001  # Simulate a 0.1% fee
        
        # Calculate profit/loss
        if position.direction == TradeDirection.LONG:
            realized_pl = market_value - cost_basis_portion - fees
        else:
            realized_pl = cost_basis_portion - market_value - fees
            
        realized_pl_percent = (realized_pl / cost_basis_portion) * 100 if cost_basis_portion > 0 else 0
        
        # Create closing trade
        close_direction = TradeDirection.SHORT if position.direction == TradeDirection.LONG else TradeDirection.LONG
        
        closing_trade = Trade(
            portfolio_id=portfolio.id,
            position_id=position.id,
            symbol=position.symbol,
            asset_class=position.asset_class,
            direction=close_direction,
            quantity=quantity,
            price=exit_price,
            status=TradeStatus.CLOSED,
            fees=fees,
            total_cost=market_value + fees,
            realized_profit_loss=realized_pl,
            realized_profit_loss_percent=realized_pl_percent,
            notes=notes,
            tags=json.dumps(tags) if tags else None,
            strategy_id=position.strategy_id
        )
        
        try:
            db.add(closing_trade)
            
            # Update portfolio
            if position.direction == TradeDirection.LONG:
                portfolio.cash_balance += (market_value - fees)
            else:
                portfolio.cash_balance += (cost_basis_portion * 0.5) + realized_pl - fees  # Return margin + profit
                
            portfolio.total_profit_loss += realized_pl
            
            # Update or delete position
            if full_close:
                # Find all open trades for this position
                open_trades = db.query(Trade).filter(
                    Trade.position_id == position.id,
                    Trade.status == TradeStatus.OPEN
                ).all()
                
                # Update trades status
                for trade in open_trades:
                    trade.status = TradeStatus.CLOSED
                    closing_trade.closing_trade_id = trade.id
                
                # Delete position
                db.delete(position)
            else:
                # Update position for partial close
                position.quantity -= quantity
                position.cost_basis -= cost_basis_portion
                position.market_value = position.quantity * position.current_price
                position.unrealized_profit_loss = position.market_value - position.cost_basis
                
                # Update trade
                opening_trade = db.query(Trade).filter(
                    Trade.position_id == position.id,
                    Trade.status == TradeStatus.OPEN
                ).first()
                
                if opening_trade:
                    # If partial close, update the opening trade
                    opening_trade.status = TradeStatus.PARTIALLY_CLOSED
                    closing_trade.closing_trade_id = opening_trade.id
            
            db.commit()
            db.refresh(closing_trade)
            
            logger.info(f"Closed {'all' if full_close else 'part of'} position ID {position_id}")
            return closing_trade
        except Exception as e:
            db.rollback()
            logger.error(f"Error closing position: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error closing position: {str(e)}"
            )
    
    @staticmethod
    async def get_position(db: Session, position_id: int, user_id: int) -> Position:
        """
        Get a specific position.
        
        Args:
            db: Database session
            position_id: ID of the position to retrieve
            user_id: ID of the user making the request
            
        Returns:
            Position instance
        """
        position = db.query(Position).filter(Position.id == position_id).first()
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Position not found"
            )
        
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, position.portfolio_id, user_id)
        
        return position
    
    @staticmethod
    async def get_portfolio_positions(db: Session, portfolio_id: int, user_id: int) -> List[Position]:
        """
        Get all positions in a portfolio.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio
            user_id: ID of the user making the request
            
        Returns:
            List of Position instances
        """
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        return db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    @staticmethod
    async def update_position_prices(db: Session, portfolio_id: int, user_id: int) -> Dict[str, Any]:
        """
        Update current prices and metrics for all positions in a portfolio.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio
            user_id: ID of the user making the request
            
        Returns:
            Dictionary with update results
        """
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        # Get all positions
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        
        if not positions:
            return {
                "status": "success",
                "message": "No positions to update",
                "updated_count": 0
            }
        
        # Group symbols by asset class for efficient market data retrieval
        symbols_by_class = {}
        for position in positions:
            if position.asset_class not in symbols_by_class:
                symbols_by_class[position.asset_class] = []
            symbols_by_class[position.asset_class].append(position.symbol)
        
        # Fetch current prices
        price_data = {}
        for asset_class, symbols in symbols_by_class.items():
            try:
                # Mock market data service call
                class_prices = await MarketDataService.get_current_prices(symbols, asset_class)
                price_data.update(class_prices)
            except Exception as e:
                logger.error(f"Error fetching prices for {asset_class}: {str(e)}")
        
        # Update positions
        updated_count = 0
        total_portfolio_value = portfolio.cash_balance
        total_unrealized_pl = 0
        
        for position in positions:
            if position.symbol in price_data:
                current_price = price_data[position.symbol]
                
                # Update position with new price
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                
                # Calculate unrealized P/L
                if position.direction == TradeDirection.LONG:
                    unrealized_pl = position.market_value - position.cost_basis
                else:
                    unrealized_pl = position.cost_basis - position.market_value
                
                position.unrealized_profit_loss = unrealized_pl
                position.unrealized_profit_loss_percent = (unrealized_pl / position.cost_basis) * 100 if position.cost_basis > 0 else 0
                
                total_portfolio_value += position.market_value
                total_unrealized_pl += unrealized_pl
                updated_count += 1
        
        # Update portfolio metrics
        portfolio.current_value = total_portfolio_value
        portfolio.daily_profit_loss = total_unrealized_pl  # Simplified, in a real app would track daily changes
        portfolio.updated_at = datetime.now()
        portfolio.last_synced_at = datetime.now()
        
        try:
            db.commit()
            logger.info(f"Updated {updated_count} positions in portfolio ID {portfolio_id}")
            
            return {
                "status": "success",
                "message": f"Updated {updated_count} positions",
                "updated_count": updated_count,
                "portfolio_value": total_portfolio_value,
                "unrealized_profit_loss": total_unrealized_pl
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating position prices: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating position prices: {str(e)}"
            )
    
    @staticmethod
    async def get_portfolio_trades(db: Session, portfolio_id: int, user_id: int, limit: int = 50, offset: int = 0) -> List[Trade]:
        """
        Get trades for a portfolio with pagination.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio
            user_id: ID of the user making the request
            limit: Maximum number of trades to return
            offset: Offset for pagination
            
        Returns:
            List of Trade instances
        """
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        return db.query(Trade).filter(
            Trade.portfolio_id == portfolio_id
        ).order_by(
            Trade.timestamp.desc()
        ).offset(offset).limit(limit).all()
    
    @staticmethod
    async def get_position_trades(db: Session, position_id: int, user_id: int) -> List[Trade]:
        """
        Get all trades for a position.
        
        Args:
            db: Database session
            position_id: ID of the position
            user_id: ID of the user making the request
            
        Returns:
            List of Trade instances
        """
        # Check access
        position = await PortfolioService.get_position(db, position_id, user_id)
        
        return db.query(Trade).filter(
            Trade.position_id == position_id
        ).order_by(
            Trade.timestamp.desc()
        ).all()
    
    @staticmethod
    async def calculate_portfolio_metrics(db: Session, portfolio_id: int, user_id: int) -> Dict[str, Any]:
        """
        Calculate performance metrics for a portfolio.
        
        Args:
            db: Database session
            portfolio_id: ID of the portfolio
            user_id: ID of the user making the request
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Check ownership
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user_id)
        
        # Get all closed trades
        closed_trades = db.query(Trade).filter(
            Trade.portfolio_id == portfolio_id,
            Trade.status == TradeStatus.CLOSED,
            Trade.realized_profit_loss.isnot(None)
        ).all()
        
        if not closed_trades:
            return {
                "status": "success",
                "message": "No closed trades to analyze",
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        
        # Calculate metrics
        winning_trades = [t for t in closed_trades if t.realized_profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.realized_profit_loss < 0]
        
        total_trades = len(closed_trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = sum(t.realized_profit_loss for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t.realized_profit_loss for t in losing_trades)) if losing_trades else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0 if total_profit == 0 else float('inf')
        
        average_win = total_profit / winning_trades_count if winning_trades_count > 0 else 0
        average_loss = total_loss / losing_trades_count if losing_trades_count > 0 else 0
        
        largest_win = max(t.realized_profit_loss for t in winning_trades) if winning_trades else 0
        largest_loss = min(t.realized_profit_loss for t in losing_trades) if losing_trades else 0
        
        # Update portfolio metrics
        portfolio.win_rate = win_rate
        portfolio.profit_factor = profit_factor
        
        try:
            db.commit()
            
            return {
                "status": "success",
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "average_win": average_win,
                "average_loss": average_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "total_trades": total_trades,
                "winning_trades": winning_trades_count,
                "losing_trades": losing_trades_count,
                "total_realized_profit": total_profit,
                "total_realized_loss": total_loss
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calculating portfolio metrics: {str(e)}"
            )
