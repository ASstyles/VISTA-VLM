# VISTA-VLM V5.0 Factory Emergency Rollback and Disaster Recovery Runbook

**Document ID:** `V5-DOC-ROLL-001`  
**Version:** 5.0.0  
**Target:** Plant Maintenance & Automation Engineers  
**Classification:** SAFETY CRITICAL OPERATIONAL PROCEDURE  

---

## 1. Emergency Rollback Triggers

An immediate line rollback to manual inspection or a previous verified model version must be executed if ANY of the following conditions occur:
1. **Critical Defect Escape**: A confirmed structural defect passes through inspection and is detected at downstream testing.
2. **False Alarm Cascade**: Overkill rate exceeds $5.0\%$ over 100 consecutive cycles, creating conveyor backlog.
3. **Heartbeat Loss / System Hang**: Inspection software process freezes or watchdog trips for $> 2,000\text{ ms}$.
4. **Physical Actuator Misalignment**: Diverter gate timing drifts, causing mechanical part collisions.

---

## 2. Immediate Line Mitigation Protocol (Level 0: Physical Bypass)

If the inspection server crashes or causes a production halt:
1. **Engage Manual Inspection Bypass**:
   Turn the physical key switch on the inspection cell panel from `AUTO (AI)` to `MANUAL BYPASS`.
   - The pneumatic diverter is mechanically latched open.
   - Parts advance to a manned visual inspection station.
   - Conveyor line continues running without software dependence.
2. **Notify Quality Supervisor**:
   Record timestamp, lot number, and serial count of parts inspected during bypass.

---

## 3. Software Model Reversion Protocol (Level 1: Rollback)

If a newly deployed checkpoint exhibits anomalous drift or unacceptable false alarms:

### Step 1: Stop Active Service
```bash
sudo systemctl stop vista-inspection.service
```

### Step 2: Revert Model Symlinks to Previous Signed Artifact
```bash
cd /opt/vista-vlm/checkpoints/
# Point active checkpoint to verified golden baseline:
ln -sfn vista_golden_v4_baseline.pth active_model.pth
ln -sfn calibration_golden_v4.json active_calibration.json
```

### Step 3: Verify Checkpoint Cryptographic Hash
```bash
sha256sum active_model.pth
# Output must match signed manifest digest in /opt/vista-vlm/manifests/trusted_checkpoints.json
```

### Step 4: Restart Inspection Daemon & Verify Health
```bash
sudo systemctl start vista-inspection.service
curl -f http://localhost:8000/health
# Confirm status is "healthy" and active model version is restored
```

---

## 4. Disaster Recovery (Total Hardware Failure)

In the event of total Industrial PC or SSD failure:
1. **Hot-Standby Swap**:
   Connect the backup pre-configured Industrial PC (kept powered in cold standby on rack slot B).
2. **Restore Signed Release Container**:
   ```bash
   docker run -d --restart=always --name vista-inspection \
     --network=host \
     --read-only \
     --user 10001:10001 \
     -v /opt/vista/checkpoints:/app/checkpoints:ro \
     -v /opt/vista/logs:/var/log/vista \
     ghcr.io/asstyles/vista-vlm:5.0.0-release
   ```
3. **Hardware Commissioning Verification**:
   Execute witness test 2 (uncalibrated check) and test 4 (watchdog interlock) before returning the cell to automated mode.

