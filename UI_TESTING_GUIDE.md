# Testing Enterprise Evaluation Features Through the UI

## Quick Start

1. **Start the server:**
   ```bash
   cd /Users/ishraq21/ragnetic
   ragnetic start-server
   ```

2. **Open the dashboard:**
   - Go to `http://localhost:8000`
   - Login with your credentials
   - Navigate to **Evaluation & Benchmarking** tab

## Step-by-Step UI Testing

### 1. **Create Test Data**
First, upload the test files I created:

- Upload `test_enterprise_eval.json` as a test set
- Upload `agents/test-enterprise-agent.yaml` as an agent config
- Upload `test_docs.txt` as source documents

### 2. **Run Enterprise Benchmark**

1. Click **"Run Benchmark"** button
2. Fill out the enhanced form:
   - **Agent**: Select `test-enterprise-agent`
   - **Test Set**: Select `test_enterprise_eval.json`
   - **Dataset ID**: Enter `enterprise-test-v1`
   - **Purpose**: Enter `Testing enterprise features`
   - **Tags**: Enter `test, enterprise, safety, pii`
3. Click **"Start Benchmark"**

### 3. **Monitor Progress**

You'll see the benchmark in the list with:
- **Status**: "running" → "completed" or "failed"
- **Cancel button**: Red X button appears for running benchmarks
- **Enterprise metrics**: New colored badges appear when complete

### 4. **View Enterprise Metrics**

When the benchmark completes, you'll see:

#### **Safety Pass Rate** (Green = Good, Yellow = Warning)
- Shows percentage of items that passed safety checks
- Good: ≥80%, Warning: <80%

#### **PII Leak Rate** (Green = Good, Red = Danger)  
- Shows percentage of items that leaked PII
- Good: ≤10%, Danger: >10%

#### **Toxicity Rate** (Green = Good, Yellow = Warning)
- Shows percentage of items with toxic language
- Good: ≤5%, Warning: >5%

#### **Multi-turn Items**
- Shows count of items with conversation history

### 5. **Test Cancel Feature**

1. Start a benchmark with a large test set
2. While it's running, click the **red X cancel button**
3. Confirm the cancellation
4. Watch the status change to "aborted"

### 6. **Test Quality Gates**

The test data includes items that should trigger quality gate failures:
- PII leakage (email/phone in answers)
- Low retrieval performance
- High toxicity

If thresholds are exceeded, the benchmark status will show **"failed"** instead of "completed".

## UI Enhancements Added

### **Enhanced Benchmark Form**
- Dataset ID field for provenance
- Purpose/Notes field for audit trail  
- Tags field for governance
- Clean, professional styling

### **Enterprise Metrics Display**
- Color-coded safety metrics (green/yellow/red)
- PII leak rate with danger indicators
- Toxicity rate with warning indicators
- Multi-turn conversation support
- Visual distinction for enterprise metrics

### **Run Lifecycle Controls**
- Cancel button for running benchmarks
- Confirmation dialog for cancellation
- Real-time status updates
- Professional loading states

## What to Look For

### **Success Indicators:**
- Form accepts all new enterprise fields
- Benchmark starts with audit metadata
- Enterprise metrics appear with proper colors
- Cancel button works for running benchmarks
- Quality gates fail benchmarks when thresholds exceeded
- All styling matches RAGnetic's professional design

### **Expected Results with Test Data:**
- **Safety Pass Rate**: ~60-80% (some items trigger safety violations)
- **PII Leak Rate**: ~20% (1 out of 5 items has PII) - should trigger quality gate
- **Toxicity Rate**: ~20% (1 out of 5 items is toxic)
- **Multi-turn Items**: 1 (one item has conversation history)
- **Status**: Should be "failed" due to PII leak rate > 10% threshold

## Key Features to Test

1. **Form Validation**: All fields are optional, form works with minimal data
2. **Enterprise Metrics**: Color coding and thresholds work correctly
3. **Cancel Functionality**: Can stop running benchmarks
4. **Quality Gates**: Benchmarks fail when thresholds exceeded
5. **Audit Trail**: Purpose, tags, and dataset ID are recorded
6. **Professional UI**: Clean, sleek design matching RAGnetic standards

The UI now provides enterprise-grade evaluation capabilities while maintaining the clean, professional appearance you requested!
