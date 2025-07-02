# ============================================================================
# scripts/download_historical_data.py - F1 Data Download Script
# ============================================================================

#!/usr/bin/env python3

"""
F1 Historical Data Download Script
Downloads F1 telemetry and lap data using FastF1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

try:
    import fastf1
    import pandas as pd
except ImportError as e:
    print(f"❌ Required package not installed: {e}")
    print("Please install with: pip install fastf1 pandas")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def download_season_data(year: int, output_dir: Path, races: Optional[List[str]] = None) -> bool:
    """Download all race data for a given season"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading data for {year} season...")
    
    try:
        # Enable caching
        cache_dir = output_dir / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        # Get race schedule
        schedule = fastf1.get_event_schedule(year)
        
        if races:
            # Filter to specific races
            schedule = schedule[schedule['EventName'].isin(races)]
        
        success_count = 0
        total_races = len(schedule)
        
        for _, event in schedule.iterrows():
            try:
                logger.info(f"Processing {event['EventName']} (Round {event['RoundNumber']})...")
                
                # Load race session
                session = fastf1.get_session(year, event['RoundNumber'], 'R')
                session.load()
                
                # Create race-specific directory
                race_dir = output_dir / f"{year}" / f"round_{event['RoundNumber']:02d}_{event['EventName'].replace(' ', '_')}"
                race_dir.mkdir(parents=True, exist_ok=True)
                
                # Save session info
                session_info = {
                    'year': year,
                    'round': event['RoundNumber'],
                    'event_name': event['EventName'],
                    'circuit': event.get('Location', 'Unknown'),
                    'date': str(event.get('Session5Date', event.get('EventDate', 'Unknown')))
                }
                
                pd.DataFrame([session_info]).to_parquet(race_dir / 'session_info.parquet')
                
                # Save lap times and results
                if hasattr(session, 'laps') and len(session.laps) > 0:
                    session.laps.to_parquet(race_dir / 'lap_times.parquet')
                    logger.info(f"Saved lap times: {len(session.laps)} laps")
                
                # Save results
                if hasattr(session, 'results') and len(session.results) > 0:
                    session.results.to_parquet(race_dir / 'results.parquet')
                
                # Save telemetry for top finishers (to manage file size)
                top_drivers = session.results.head(10)['Abbreviation'].tolist() if hasattr(session, 'results') else session.drivers[:10]
                
                for driver in top_drivers:
                    try:
                        driver_laps = session.laps.pick_driver(driver)
                        if len(driver_laps) > 0:
                            # Get telemetry for fastest lap
                            fastest_lap = driver_laps.pick_fastest()
                            if fastest_lap is not None:
                                telemetry = fastest_lap.get_telemetry()
                                if len(telemetry) > 0:
                                    filename = f'telemetry_{driver}.parquet'
                                    telemetry.to_parquet(race_dir / filename)
                                    logger.debug(f"Saved telemetry for {driver}")
                    except Exception as e:
                        logger.warning(f"Could not save telemetry for {driver}: {e}")
                
                success_count += 1
                logger.info(f"✅ Completed {event['EventName']} ({success_count}/{total_races})")
                
            except Exception as e:
                logger.error(f"❌ Error processing {event['EventName']}: {e}")
                continue
        
        logger.info(f"Download complete! Successfully processed {success_count}/{total_races} races")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error downloading season data: {e}")
        return False


def download_sample_data(output_dir: Path) -> bool:
    """Download minimal sample data for testing"""
    
    logger = logging.getLogger(__name__)
    logger.info("Downloading sample F1 data for testing...")
    
    try:
        # Enable caching
        cache_dir = output_dir / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        # Download 2024 Bahrain GP as sample
        logger.info("Loading 2024 Bahrain Grand Prix...")
        session = fastf1.get_session(2024, 'Bahrain', 'R')
        session.load()
        
        sample_dir = output_dir / 'sample'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save basic lap data
        session.laps.to_parquet(sample_dir / 'sample_lap_times.parquet')
        logger.info(f"Saved sample lap times: {len(session.laps)} laps")
        
        # Save telemetry for top 3 drivers
        top_drivers = ['VER', 'LEC', 'SAI']  # Typical top performers
        
        for driver in top_drivers:
            try:
                driver_laps = session.laps.pick_driver(driver)
                if len(driver_laps) > 0:
                    fastest_lap = driver_laps.pick_fastest()
                    if fastest_lap is not None:
                        telemetry = fastest_lap.get_telemetry()
                        if len(telemetry) > 0:
                            filename = f'sample_telemetry_{driver}.parquet'
                            telemetry.to_parquet(sample_dir / filename)
                            logger.info(f"Saved sample telemetry for {driver}")
            except Exception as e:
                logger.warning(f"Could not download sample data for {driver}: {e}")
        
        logger.info("✅ Sample data download complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading sample data: {e}")
        logger.info("This might be due to network issues or FastF1 API limits")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download F1 historical data using FastF1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download sample data for testing
  python download_historical_data.py --sample
  
  # Download full 2024 season
  python download_historical_data.py --years 2024
  
  # Download specific years
  python download_historical_data.py --years 2022-2024
  
  # Download specific races
  python download_historical_data.py --years 2024 --races "Bahrain" "Monaco"
        """
    )
    
    parser.add_argument(
        '--years',
        help='Year or year range (e.g., 2024 or 2022-2024)',
        default=None
    )
    
    parser.add_argument(
        '--races',
        nargs='+',
        help='Specific race names to download',
        default=None
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/historical'),
        help='Output directory (default: data/historical)'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Download sample data only (for testing)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Download data
    success = False
    
    if args.sample:
        success = download_sample_data(args.output)
    elif args.years:
        if '-' in args.years:
            start_year, end_year = map(int, args.years.split('-'))
            years = range(start_year, end_year + 1)
        else:
            years = [int(args.years)]
        
        for year in years:
            year_success = download_season_data(year, args.output, args.races)
            success = success or year_success
    else:
        logger.error("Please specify --years or --sample")
        parser.print_help()
        sys.exit(1)
    
    if success:
        logger.info("🎉 Data download completed successfully!")
        logger.info(f"Data saved to: {args.output.absolute()}")
    else:
        logger.error("❌ Data download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
