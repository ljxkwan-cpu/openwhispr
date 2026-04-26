const fs = require("fs");
const fsPromises = require("fs").promises;
const path = require("path");
const { spawn } = require("child_process");
const { pipeline } = require("stream/promises");
const debugLogger = require("./debugLogger");
const {
  downloadFile,
  createDownloadSignal,
  cleanupStaleDownloads,
  checkDiskSpace,
} = require("./downloadUtils");
const Qwen3AsrServerManager = require("./qwen3AsrServer");
const { getModelsDirForService } = require("./modelDirUtils");

const modelRegistryData = require("../models/modelRegistryData.json");

function getQwen3AsrModelConfig(modelName) {
  const modelInfo = modelRegistryData.qwen3AsrModels[modelName];
  if (!modelInfo) return null;
  return {
    url: modelInfo.downloadUrl,
    size: modelInfo.sizeMb * 1_000_000,
    extractDir: modelInfo.extractDir,
  };
}

function getValidModelNames() {
  return Object.keys(modelRegistryData.qwen3AsrModels);
}

class Qwen3AsrManager {
  constructor() {
    this.currentDownloadProcess = null;
    this.isInitialized = false;
    this.serverManager = new Qwen3AsrServerManager();
  }

  getModelsDir() {
    return getModelsDirForService("qwen3-asr");
  }

  validateModelName(modelName) {
    const validModels = getValidModelNames();
    if (!validModels.includes(modelName)) {
      throw new Error(
        `Invalid Qwen3-ASR model: ${modelName}. Valid models: ${validModels.join(", ")}`
      );
    }
    return true;
  }

  getModelPath(modelName) {
    this.validateModelName(modelName);
    return path.join(this.getModelsDir(), modelName);
  }

  async initializeAtStartup(settings = {}) {
    const startTime = Date.now();

    try {
      this.isInitialized = true;

      await cleanupStaleDownloads(this.getModelsDir());

      const { localTranscriptionProvider, qwen3AsrModel } = settings;

      if (
        localTranscriptionProvider === "qwen3" &&
        qwen3AsrModel &&
        this.serverManager.isAvailable()
      ) {
        if (this.serverManager.isModelDownloaded(qwen3AsrModel)) {
          debugLogger.info("Pre-warming qwen3-asr server", { model: qwen3AsrModel });

          try {
            const serverStartTime = Date.now();
            await this.serverManager.startServer(qwen3AsrModel);
            debugLogger.info("Qwen3-ASR server pre-warmed successfully", {
              model: qwen3AsrModel,
              startupTimeMs: Date.now() - serverStartTime,
            });
          } catch (err) {
            debugLogger.warn("Qwen3-ASR server pre-warm failed (will start on first use)", {
              error: err.message,
              model: qwen3AsrModel,
            });
          }
        }
      }
    } catch (error) {
      debugLogger.warn("Qwen3-ASR initialization error", { error: error.message });
      this.isInitialized = true;
    }

    debugLogger.info("Qwen3-ASR initialization complete", {
      totalTimeMs: Date.now() - startTime,
      binaryAvailable: this.serverManager.isAvailable(),
    });
  }

  async checkInstallation() {
    const binaryPath = this.serverManager.getBinaryPath();
    if (!binaryPath) {
      return { installed: false, working: false };
    }

    return {
      installed: true,
      working: this.serverManager.isAvailable(),
      path: binaryPath,
    };
  }

  async startServer(modelName) {
    this.validateModelName(modelName);
    return this.serverManager.startServer(modelName);
  }

  async stopServer() {
    await this.serverManager.stopServer();
  }

  getServerStatus() {
    return this.serverManager.getServerStatus();
  }

  async transcribeLocalQwen3Asr(audioBlob, options = {}) {
    debugLogger.logSTTPipeline("transcribeLocalQwen3Asr - start", {
      options,
      audioBlobSize: audioBlob?.byteLength || audioBlob?.size || 0,
      serverAvailable: this.serverManager.isAvailable(),
    });

    if (!this.serverManager.isAvailable()) {
      throw new Error(
        "sherpa-onnx binary not found. Please ensure the app is installed correctly."
      );
    }

    const model = options.model || "qwen3-asr-0.6b";

    if (!this.serverManager.isModelDownloaded(model)) {
      throw new Error(
        `Qwen3-ASR model "${model}" not downloaded. Please download it from Settings.`
      );
    }

    let audioBuffer;
    if (Buffer.isBuffer(audioBlob)) {
      audioBuffer = audioBlob;
    } else if (ArrayBuffer.isView(audioBlob)) {
      audioBuffer = Buffer.from(audioBlob.buffer, audioBlob.byteOffset, audioBlob.byteLength);
    } else if (audioBlob instanceof ArrayBuffer) {
      audioBuffer = Buffer.from(audioBlob);
    } else if (typeof audioBlob === "string") {
      audioBuffer = Buffer.from(audioBlob, "base64");
    } else if (audioBlob && audioBlob.buffer && typeof audioBlob.byteLength === "number") {
      audioBuffer = Buffer.from(audioBlob.buffer, audioBlob.byteOffset || 0, audioBlob.byteLength);
    } else {
      throw new Error(`Unsupported audio data type: ${typeof audioBlob}`);
    }

    if (!audioBuffer || audioBuffer.length === 0) {
      throw new Error("Audio buffer is empty - no audio data received");
    }

    const startTime = Date.now();
    const language = options.language || "auto";
    const result = await this.serverManager.transcribe(audioBuffer, { modelName: model, language });
    const elapsed = Date.now() - startTime;

    debugLogger.logSTTPipeline("transcribeLocalQwen3Asr - completed", {
      elapsed,
      textLength: result.text?.length || 0,
    });

    return this._parseResult(result);
  }

  _parseResult(output) {
    if (!output || !output.text) {
      return { success: false, message: "No audio detected" };
    }

    const text = output.text.trim();

    if (!text || text.length === 0) {
      return { success: false, message: "No audio detected" };
    }

    return { success: true, text };
  }

  async downloadQwen3AsrModel(modelName, progressCallback = null) {
    this.validateModelName(modelName);
    const modelConfig = getQwen3AsrModelConfig(modelName);

    const modelPath = this.getModelPath(modelName);
    const modelsDir = this.getModelsDir();

    await fsPromises.mkdir(modelsDir, { recursive: true });

    if (this.serverManager.isModelDownloaded(modelName)) {
      return { model: modelName, downloaded: true, path: modelPath, success: true };
    }

    const spaceCheck = await checkDiskSpace(modelsDir, modelConfig.size * 2.5);
    if (!spaceCheck.ok) {
      throw new Error(
        `Not enough disk space to download and extract model. Need ~${Math.round((modelConfig.size * 2.5) / 1_000_000)}MB, ` +
          `only ${Math.round(spaceCheck.availableBytes / 1_000_000)}MB available.`
      );
    }

    const archivePath = path.join(modelsDir, `${modelName}.tar.bz2`);
    const { signal, abort } = createDownloadSignal();
    this.currentDownloadProcess = { abort };

    try {
      let archiveReady = false;
      try {
        const stats = await fsPromises.stat(archivePath);
        if (stats.size > 0) {
          archiveReady = true;
        }
      } catch {}

      if (!archiveReady) {
        await downloadFile(modelConfig.url, archivePath, {
          timeout: 600000,
          signal,
          onProgress: (downloadedBytes, totalBytes) => {
            if (progressCallback) {
              progressCallback({
                type: "progress",
                model: modelName,
                downloaded_bytes: downloadedBytes,
                total_bytes: totalBytes,
                percentage: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
              });
            }
          },
        });
      }

      if (progressCallback) {
        progressCallback({ type: "installing", model: modelName, percentage: 100 });
      }

      const MAX_EXTRACT_RETRIES = 2;
      for (let attempt = 1; attempt <= MAX_EXTRACT_RETRIES; attempt++) {
        try {
          await this._extractModel(archivePath, modelName);
          break;
        } catch (extractError) {
          if (attempt >= MAX_EXTRACT_RETRIES) {
            const err = new Error(`Model installation failed: ${extractError.message}`);
            err.code = "EXTRACTION_FAILED";
            throw err;
          }
        }
      }
      await fsPromises.unlink(archivePath).catch(() => {});

      if (progressCallback) {
        progressCallback({ type: "complete", model: modelName, percentage: 100 });
      }

      if (this.serverManager.isAvailable()) {
        this.serverManager.startServer(modelName).catch((err) => {
          debugLogger.warn("Post-download server pre-warm failed (non-fatal)", {
            error: err.message,
            model: modelName,
          });
        });
      }

      return { model: modelName, downloaded: true, path: modelPath, success: true };
    } catch (error) {
      if (error.isAbort) {
        await fsPromises.unlink(archivePath).catch(() => {});
        throw new Error("Download interrupted by user");
      }
      throw error;
    } finally {
      this.currentDownloadProcess = null;
    }
  }

  async _extractModel(archivePath, modelName) {
    const modelsDir = this.getModelsDir();
    const modelConfig = getQwen3AsrModelConfig(modelName);
    const extractDir = path.join(modelsDir, `temp-extract-${modelName}`);

    try {
      await fsPromises.mkdir(extractDir, { recursive: true });
      await this._runTarExtract(archivePath, extractDir);

      const extractedDir = path.join(extractDir, modelConfig.extractDir);
      const targetDir = this.getModelPath(modelName);

      if (fs.existsSync(extractedDir)) {
        if (fs.existsSync(targetDir)) {
          await fsPromises.rm(targetDir, { recursive: true, force: true });
        }
        await fsPromises.rename(extractedDir, targetDir);
      } else {
        const entries = await fsPromises.readdir(extractDir);
        let modelDir = null;

        for (const entry of entries) {
          const entryPath = path.join(extractDir, entry);
          const stat = await fsPromises.stat(entryPath);
          if (stat.isDirectory() && entry.includes("qwen3")) {
            modelDir = entry;
            break;
          }
        }

        if (modelDir) {
          if (fs.existsSync(targetDir)) {
            await fsPromises.rm(targetDir, { recursive: true, force: true });
          }
          await fsPromises.rename(path.join(extractDir, modelDir), targetDir);
        } else {
          throw new Error(
            `Could not find model directory in extracted archive. ` +
              `Expected "${modelConfig.extractDir}", found: [${entries.join(", ")}]`
          );
        }
      }

      const requiredFiles = [
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        path.join("tokenizer", "vocab.json"),
      ];
      const missing = requiredFiles.filter((f) => !fs.existsSync(path.join(targetDir, f)));
      if (missing.length > 0) {
        throw new Error(`Extracted model is missing required files: ${missing.join(", ")}`);
      }

      await fsPromises.rm(extractDir, { recursive: true, force: true });
    } catch (error) {
      try {
        await fsPromises.rm(extractDir, { recursive: true, force: true });
      } catch {}
      throw error;
    }
  }

  async _runTarExtract(archivePath, extractDir) {
    try {
      await this._runSystemTar(archivePath, extractDir);
      return;
    } catch (err) {
      debugLogger.debug("System tar failed, falling back to JS extraction", {
        error: err.message,
      });
    }

    const unbzip2 = require("unbzip2-stream");
    const tar = require("tar");
    await pipeline(fs.createReadStream(archivePath), unbzip2(), tar.x({ cwd: extractDir }));
  }

  _runSystemTar(archivePath, extractDir) {
    return new Promise((resolve, reject) => {
      const tarProcess = spawn("tar", ["-xjf", archivePath, "-C", extractDir], {
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stderr = "";

      tarProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      tarProcess.on("close", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`tar extraction failed with code ${code}: ${stderr}`));
        }
      });

      tarProcess.on("error", (err) => {
        reject(new Error(`Failed to start tar process: ${err.message}`));
      });
    });
  }

  async cancelDownload() {
    if (this.currentDownloadProcess) {
      this.currentDownloadProcess.abort();
      this.currentDownloadProcess = null;
      return { success: true, message: "Download cancelled" };
    }
    return { success: false, error: "No active download to cancel" };
  }

  async checkModelStatus(modelName) {
    const modelPath = this.getModelPath(modelName);

    if (this.serverManager.isModelDownloaded(modelName)) {
      try {
        const encoderPath = path.join(modelPath, "encoder.int8.onnx");
        const stats = fs.statSync(encoderPath);
        return {
          model: modelName,
          downloaded: true,
          path: modelPath,
          size_bytes: stats.size,
          size_mb: Math.round(stats.size / (1024 * 1024)),
          success: true,
        };
      } catch {
        return { model: modelName, downloaded: false, success: true };
      }
    }

    return { model: modelName, downloaded: false, success: true };
  }

  async listQwen3AsrModels() {
    const models = getValidModelNames();
    const modelInfo = [];

    for (const model of models) {
      const status = await this.checkModelStatus(model);
      modelInfo.push(status);
    }

    return {
      models: modelInfo,
      cache_dir: this.getModelsDir(),
      success: true,
    };
  }

  async deleteQwen3AsrModel(modelName) {
    const modelPath = this.getModelPath(modelName);

    if (fs.existsSync(modelPath)) {
      try {
        fs.rmSync(modelPath, { recursive: true, force: true });
        return { model: modelName, deleted: true, success: true };
      } catch (error) {
        return { model: modelName, deleted: false, error: error.message, success: false };
      }
    }

    return { model: modelName, deleted: false, error: "Model not found", success: false };
  }

  async getDiagnostics() {
    return {
      platform: process.platform,
      arch: process.arch,
      sherpaOnnx: {
        available: this.serverManager.isAvailable(),
        path: this.serverManager.getBinaryPath(),
      },
      modelsDir: this.getModelsDir(),
      models: this.serverManager.isAvailable()
        ? getValidModelNames().filter((m) => this.serverManager.isModelDownloaded(m))
        : [],
    };
  }
}

module.exports = Qwen3AsrManager;
