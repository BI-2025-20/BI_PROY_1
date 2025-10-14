// ============================================
// Python Bridge Service (Comunicación con modelo ML)
// src/services/python-bridge.service.ts
// ============================================

import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

interface PythonPredictionResult {
  label: number;
  probability: number;
  class_probabilities: { [key: string]: number };
  text: string;
}

interface PythonResponse {
  ok: boolean;
  result?: PythonPredictionResult[];
  error?: string;
}

interface PythonRetrainMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  precision_per_class: { [key: string]: number };
  recall_per_class: { [key: string]: number };
  f1_score_per_class: { [key: string]: number };
  model_version: string;
  training_samples?: number;
  validation_samples?: number;
}

@Injectable()
export class PythonBridgeService implements OnModuleInit {
  private readonly logger = new Logger(PythonBridgeService.name);
  
  // Rutas correctas apuntando a src/
  private readonly pythonScriptPath = path.join(process.cwd(), 'src', 'python_scripts');
  private readonly modelPath = path.join(process.cwd(), 'src', 'ml_models', 'modelo_reentrenado.joblib');
  
  // Comando Python según el sistema operativo
  private readonly pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  async onModuleInit() {
    this.logger.log(`📁 CWD: ${process.cwd()}`);
    this.logger.log(`🐍 Python scripts: ${this.pythonScriptPath}`);
    this.logger.log(`🤖 Model path: ${this.modelPath}`);
    this.logger.log(`💻 Python command: ${this.pythonCmd}`);
    
    // Verificar que existan los archivos
    const predictPath = path.join(this.pythonScriptPath, 'predict.py');
    const retrainPath = path.join(this.pythonScriptPath, 'retrain.py');
    
    if (fs.existsSync(predictPath)) {
      this.logger.log(`✅ predict.py encontrado`);
    } else {
      this.logger.error(`❌ predict.py NO encontrado en: ${predictPath}`);
    }
    
    if (fs.existsSync(retrainPath)) {
      this.logger.log(`✅ retrain.py encontrado`);
    } else {
      this.logger.error(`❌ retrain.py NO encontrado en: ${retrainPath}`);
    }
    
    if (fs.existsSync(this.modelPath)) {
      this.logger.log(`✅ modelo_reentrenado.joblib encontrado`);
    } else {
      this.logger.warn(`⚠️  modelo_reentrenado.joblib NO encontrado en: ${this.modelPath}`);
    }
  }

  /**
   * Realiza predicciones usando el modelo Python
   */
  async predict(processedTexts: string[]): Promise<PythonPredictionResult[]> {
    const scriptPath = path.join(this.pythonScriptPath, 'predict.py');
    
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`predict.py not found at: ${scriptPath}`);
    }

    return new Promise((resolve, reject) => {
      this.logger.debug(`Ejecutando: ${this.pythonCmd} ${scriptPath}`);
      
      const pythonProcess = spawn(this.pythonCmd, [
        scriptPath,
        JSON.stringify(processedTexts)
      ]);

      let dataString = '';
      let errorString = '';

      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorString += data.toString();
        this.logger.debug(`Python: ${data.toString().trim()}`);
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`❌ Python exit code ${code}`);
          this.logger.error(`Error: ${errorString}`);
          reject(new Error(`Python script failed with code ${code}: ${errorString}`));
        } else {
          try {
            this.logger.debug(`Respuesta Python: ${dataString}`);
            const response: PythonResponse = JSON.parse(dataString);
            
            if (response.ok && response.result) {
              this.logger.log(`✅ Predicciones: ${response.result.length} resultados`);
              resolve(response.result);
            } else {
              const errorMsg = response.error || 'Unknown error from Python';
              this.logger.error(`❌ Python error: ${errorMsg}`);
              reject(new Error(errorMsg));
            }
          } catch (error) {
            this.logger.error(`❌ Error parseando JSON: ${dataString}`);
            reject(new Error(`Failed to parse Python output: ${dataString}`));
          }
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error(`❌ Error spawning Python: ${error.message}`);
        reject(new Error(`Failed to spawn Python: ${error.message}`));
      });
    });
  }

  /**
   * Re-entrena el modelo usando nuevos datos
   */
  async retrain(texts: string[], labels: number[]): Promise<PythonRetrainMetrics> {
    const scriptPath = path.join(this.pythonScriptPath, 'retrain.py');
    
    if (!fs.existsSync(scriptPath)) {
      throw new Error(`retrain.py not found at: ${scriptPath}`);
    }

    return new Promise((resolve, reject) => {
      this.logger.debug(`Ejecutando: ${this.pythonCmd} ${scriptPath}`);
      
      const pythonProcess = spawn(this.pythonCmd, [
        scriptPath,
        JSON.stringify({ texts, labels })
      ]);

      let dataString = '';
      let errorString = '';

      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorString += data.toString();
        this.logger.debug(`Python: ${data.toString().trim()}`);
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          this.logger.error(`❌ Python exit code ${code}`);
          this.logger.error(`Error: ${errorString}`);
          reject(new Error(`Python retrain failed with code ${code}: ${errorString}`));
        } else {
          try {
            this.logger.debug(`Respuesta Python: ${dataString}`);
            const metrics: PythonRetrainMetrics = JSON.parse(dataString);
            this.logger.log(`✅ Re-entrenamiento OK. F1: ${metrics.f1_score}`);
            resolve(metrics);
          } catch (error) {
            this.logger.error(`❌ Error parseando JSON: ${dataString}`);
            reject(new Error(`Failed to parse retrain output: ${dataString}`));
          }
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error(`❌ Error spawning Python: ${error.message}`);
        reject(new Error(`Failed to spawn Python: ${error.message}`));
      });
    });
  }
}