#!/usr/bin/env bash
# Start the campaign now and keep it going: cron restarts the runner every
# 15 minutes (after a reboot or a crash), flock keeps it to one at a time.
CAMPAIGN=$(cd "$(dirname "$0")" && pwd)
# GMX, when set, is passed on to every job (e.g. a double-precision build)
env=${GMX:+GMX=$GMX }
line="*/15 * * * * ${env}flock -n $CAMPAIGN/.lock $CAMPAIGN/runner.sh >> $CAMPAIGN/runner.log 2>&1 # oniom-campaign"
( crontab -l 2>/dev/null | grep -v "oniom-campaign"; echo "$line" ) | crontab -
crontab -l | grep oniom-campaign
nohup flock -n "$CAMPAIGN/.lock" "$CAMPAIGN/runner.sh" >> "$CAMPAIGN/runner.log" 2>&1 &
