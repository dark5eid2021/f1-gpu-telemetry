#!/bin/bash

# F1 GPU Telemetry System - Automated Data Update Script
# Updates historical F1 data on a scheduled basis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}============================================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}============================================================================${NC}"
}

# Configuration
DATA_DIR="${DATA_DIR:-data/historical}"
CACHE_DIR="${CACHE_DIR:-data/cache}"
LOG_DIR="${LOG_DIR:-logs}"
CURRENT_YEAR=$(date +%Y)
UPDATE_INTERVAL="${UPDATE_INTERVAL:-weekly}"  # daily, weekly, monthly
MAX_RETRIES=3
LOCK_FILE="/tmp/f1-data-update.lock"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --year)
            TARGET_YEAR="$2"
            shift 2
            ;;
        --race)
            TARGET_RACE="$2"
            shift 2
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

show_help() {
    echo "F1 Data Update Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --interval FREQ    Update frequency: daily, weekly, monthly (default: weekly)"
    echo "  --year YEAR        Update specific year only"
    echo "  --race RACE        Update specific race only"
    echo "  --force            Force update even if recent data exists"
    echo "  --dry-run          Show what would be updated without doing it"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DATA_DIR           Directory for historical data (default: data/historical)"
    echo "  CACHE_DIR          Directory for FastF1 cache (default: data/cache)"
    echo "  LOG_DIR            Directory for logs (default: logs)"
    echo "  UPDATE_INTERVAL    Default update interval (default: weekly)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Regular weekly update"
    echo "  $0 --interval daily             # Daily updates"
    echo "  $0 --year 2024 --force          # Force update 2024 season"
    echo "  $0 --race 'Monaco' --dry-run    # Check what Monaco data needs updating"
}

# Check if another update is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_error "Another data update is already running (PID: $pid)"
            exit 1
        else
            print_warning "Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
    trap cleanup EXIT
}

cleanup() {
    rm -f "$LOCK_FILE"
}

# Setup directories and logging
setup_environment() {
    print_header "🏗️ Setting Up Environment"
    
    # Create directories
    mkdir -p "$DATA_DIR" "$CACHE_DIR" "$LOG_DIR"
    
    # Setup logging
    local log_file="$LOG_DIR/data-update-$(date +%Y%m%d-%H%M%S).log"
    exec 1> >(tee -a "$log_file")
    exec 2> >(tee -a "$log_file" >&2)
    
    print_success "Environment setup complete"
    print_status "Log file: $log_file"
}

# Check if Python environment is ready
check_python_environment() {
    print_header "🐍 Checking Python Environment"
    
    # Check if we're in a conda environment
    if [[ "$CONDA_DEFAULT_ENV" != "" ]] && [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
        print_success "Using conda environment: $CONDA_DEFAULT_ENV"
    elif [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_warning "No Python environment detected, using system Python"
    fi
    
    # Check required packages
    local missing_packages=()
    
    for package in fastf1 pandas numpy; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_status "Installing missing packages..."
        pip install "${missing_packages[@]}"
    fi
    
    print_success "Python environment ready"
}

# Get list of current F1 season races
get_current_races() {
    print_status "Fetching current F1 season information..."
    
    python3 -c "
import fastf1
import sys
from datetime import datetime

try:
    # Get current season schedule
    schedule = fastf1.get_event_schedule($CURRENT_YEAR)
    current_date = datetime.now()
    
    # Find races that have already happened
    completed_races = []
    upcoming_races = []
    
    for _, event in schedule.iterrows():
        event_date = event.get('Session5Date', event.get('EventDate'))
        if event_date and event_date < current_date:
            completed_races.append(event['EventName'])
        else:
            upcoming_races.append(event['EventName'])
    
    print(f'COMPLETED_RACES={\',\'.join(completed_races)}')
    print(f'UPCOMING_RACES={\',\'.join(upcoming_races)}')
    print(f'TOTAL_RACES={len(schedule)}')
    
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" > /tmp/f1_season_info.txt

    if [ $? -eq 0 ]; then
        source /tmp/f1_season_info.txt
        print_success "Found $TOTAL_RACES races in $CURRENT_YEAR season"
        print_status "Completed races: $(echo $COMPLETED_RACES | tr ',' ' ')"
        rm -f /tmp/f1_season_info.txt
    else
        print_error "Failed to fetch season information"
        return 1
    fi
}

# Check what data needs updating
check_data_freshness() {
    print_header "📊 Checking Data Freshness"
    
    local needs_update=false
    local update_reasons=()
    
    # Check if we have current year data
    if [ ! -d "$DATA_DIR/$CURRENT_YEAR" ]; then
        needs_update=true
        update_reasons+=("No data for current year ($CURRENT_YEAR)")
    fi
    
    # Check data age based on update interval
    local cutoff_days
    case "$UPDATE_INTERVAL" in
        daily) cutoff_days=1 ;;
        weekly) cutoff_days=7 ;;
        monthly) cutoff_days=30 ;;
        *) cutoff_days=7 ;;
    esac
    
    # Find files older than cutoff
    local old_files=$(find "$DATA_DIR" -name "*.parquet" -mtime +$cutoff_days 2>/dev/null | wc -l)
    if [ "$old_files" -gt 0 ]; then
        needs_update=true
        update_reasons+=("$old_files files older than $cutoff_days days")
    fi
    
    # Check if latest completed race is missing
    if [ -n "$COMPLETED_RACES" ]; then
        local latest_race=$(echo "$COMPLETED_RACES" | tr ',' '\n' | tail -1)
        local race_file=$(find "$DATA_DIR/$CURRENT_YEAR" -name "*${latest_race// /_}*" 2>/dev/null | head -1)
        if [ -z "$race_file" ]; then
            needs_update=true
            update_reasons+=("Missing latest race: $latest_race")
        fi
    fi
    
    # Force update if requested
    if [ "$FORCE_UPDATE" = true ]; then
        needs_update=true
        update_reasons+=("Force update requested")
    fi
    
    if [ "$needs_update" = true ]; then
        print_warning "Data update needed:"
        for reason in "${update_reasons[@]}"; do
            print_status "  - $reason"
        done
        return 0
    else
        print_success "Data is up to date"
        return 1
    fi
}

# Download latest data
update_data() {
    print_header "📥 Updating F1 Data"
    
    if [ "$DRY_RUN" = true ]; then
        print_status "DRY RUN: Would update data but not actually downloading"
        return 0
    fi
    
    local retry_count=0
    local download_success=false
    
    while [ $retry_count -lt $MAX_RETRIES ] && [ "$download_success" = false ]; do
        if [ $retry_count -gt 0 ]; then
            print_status "Retry attempt $retry_count of $MAX_RETRIES..."
            sleep 5
        fi
        
        # Determine what to download
        local download_args=""
        
        if [ -n "$TARGET_YEAR" ]; then
            download_args="--years $TARGET_YEAR"
        elif [ -n "$TARGET_RACE" ]; then
            download_args="--years $CURRENT_YEAR --races '$TARGET_RACE'"
        else
            # Default: update current year and previous year
            local prev_year=$((CURRENT_YEAR - 1))
            download_args="--years $prev_year-$CURRENT_YEAR"
        fi
        
        print_status "Running: python scripts/download_historical_data.py $download_args --output $DATA_DIR"
        
        if python scripts/download_historical_data.py $download_args --output "$DATA_DIR"; then
            download_success=true
            print_success "Data download completed successfully"
        else
            retry_count=$((retry_count + 1))
            print_warning "Download failed (attempt $retry_count)"
        fi
    done
    
    if [ "$download_success" = false ]; then
        print_error "Data download failed after $MAX_RETRIES attempts"
        return 1
    fi
    
    return 0
}

# Update data statistics and metadata
update_metadata() {
    print_header "📋 Updating Metadata"
    
    local metadata_file="$DATA_DIR/metadata.json"
    local file_count=$(find "$DATA_DIR" -name "*.parquet" | wc -l)
    local total_size=$(du -sh "$DATA_DIR" | cut -f1)
    
    cat > "$metadata_file" << EOF
{
    "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "update_interval": "$UPDATE_INTERVAL",
    "total_files": $file_count,
    "total_size": "$total_size",
    "current_year": $CURRENT_YEAR,
    "data_directory": "$DATA_DIR",
    "cache_directory": "$CACHE_DIR"
}
EOF
    
    print_success "Metadata updated: $metadata_file"
    print_status "Total files: $file_count"
    print_status "Total size: $total_size"
}

# Clean up old data and cache
cleanup_old_data() {
    print_header "🧹 Cleaning Up Old Data"
    
    # Remove data older than 3 years (keep current + 2 previous years)
    local cutoff_year=$((CURRENT_YEAR - 2))
    
    for year_dir in "$DATA_DIR"/*; do
        if [ -d "$year_dir" ]; then
            local year=$(basename "$year_dir")
            if [[ "$year" =~ ^[0-9]{4}$ ]] && [ "$year" -lt "$cutoff_year" ]; then
                if [ "$DRY_RUN" = true ]; then
                    print_status "DRY RUN: Would remove old data: $year_dir"
                else
                    print_status "Removing old data: $year_dir"
                    rm -rf "$year_dir"
                fi
            fi
        fi
    done
    
    # Clean FastF1 cache (keep last 30 days)
    if [ -d "$CACHE_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            print_status "DRY RUN: Would clean cache older than 30 days"
        else
            find "$CACHE_DIR" -type f -mtime +30 -delete 2>/dev/null || true
            print_success "Cache cleaned"
        fi
    fi
    
    # Clean old log files (keep last 14 days)
    if [ -d "$LOG_DIR" ]; then
        if [ "$DRY_RUN" = true ]; then
            print_status "DRY RUN: Would clean logs older than 14 days"
        else
            find "$LOG_DIR" -name "data-update-*.log" -mtime +14 -delete 2>/dev/null || true
            print_success "Old logs cleaned"
        fi
    fi
}

# Send notification about update status
send_notification() {
    local status="$1"
    local message="$2"
    
    # This could be extended to send Slack/email notifications
    print_status "Notification: $status - $message"
    
    # Log to system log if available
    if command -v logger &> /dev/null; then
        logger -t "f1-data-update" "$status: $message"
    fi
    
    # Could add webhook notifications here
    # curl -X POST "$WEBHOOK_URL" -d "{\"text\":\"F1 Data Update: $status - $message\"}"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    print_header "🏎️ F1 Data Update - $(date)"
    print_status "Update interval: $UPDATE_INTERVAL"
    print_status "Target year: ${TARGET_YEAR:-$CURRENT_YEAR}"
    print_status "Target race: ${TARGET_RACE:-all}"
    
    # Setup
    check_lock
    setup_environment
    check_python_environment
    
    # Get current season info
    if ! get_current_races; then
        send_notification "ERROR" "Failed to fetch current season information"
        exit 1
    fi
    
    # Check if update is needed
    if check_data_freshness; then
        print_status "Proceeding with data update..."
        
        # Perform update
        if update_data; then
            update_metadata
            cleanup_old_data
            
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            
            print_success "Data update completed in ${duration}s"
            send_notification "SUCCESS" "Data update completed successfully in ${duration}s"
        else
            send_notification "ERROR" "Data update failed"
            exit 1
        fi
    else
        print_success "No update needed"
        send_notification "INFO" "Data is up to date, no update needed"
    fi
    
    print_header "🎉 Update Complete"
}

# Run main function
main "$@"

---
# Cron job examples (add to crontab with: crontab -e)

# Daily update at 2 AM
# 0 2 * * * /path/to/f1-gpu-telemetry/scripts/update-data.sh --interval daily >> /var/log/f1-data-update.log 2>&1

# Weekly update on Mondays at 3 AM  
# 0 3 * * 1 /path/to/f1-gpu-telemetry/scripts/update-data.sh --interval weekly >> /var/log/f1-data-update.log 2>&1

# Monthly update on 1st of month at 4 AM
# 0 4 1 * * /path/to/f1-gpu-telemetry/scripts/update-data.sh --interval monthly >> /var/log/f1-data-update.log 2>&1

# Race weekend update (Friday-Sunday during F1 season)
# 0 */6 * * 5-0 /path/to/f1-gpu-telemetry/scripts/update-data.sh --interval daily --force >> /var/log/f1-data-update.log 2>&1