// ============================================
// 5. ML Service (Orquestador de ML)
// src/services/ml.service.ts
// ============================================

import { Injectable, Logger } from "@nestjs/common";
import { PipelineService } from "./pipeline.service";
import { PythonBridgeService } from "./python-bridge.service";
import { PredictionResultDto } from "../dto/prediction.dto";  // Ajusta la ruta si es necesario
import { RetrainResponseDto } from "../dto/retrain.dto";      // Ajusta la ruta si es necesario

@Injectable()
export class MLService {
  private readonly logger = new Logger(MLService.name);
  
  constructor(
    private readonly pipelineService: PipelineService,
    private readonly pythonBridge: PythonBridgeService,
  ) {}

  /**
   * Realiza predicciones para una lista de textos
   */
  async predictBatch(texts: string[]): Promise<PredictionResultDto[]> {
    try {
      // 1. Preprocesar textos (solo limpieza básica)
      this.logger.log(`📝 Preprocesando ${texts.length} textos...`);
      const processedTexts = this.pipelineService.preprocessBatch(texts);

      // 2. Obtener predicciones del modelo Python
      this.logger.log('🤖 Ejecutando predicciones...');
      const predictions = await this.pythonBridge.predict(processedTexts);

      // 3. Formatear resultados
      const results: PredictionResultDto[] = predictions.map((pred, idx) => ({
        predicted_label: pred.label,
        probability: pred.probability,
        textos: texts[idx],
        class_probabilities: pred.class_probabilities,
      }));

      this.logger.log(`✅ Predicciones completadas: ${results.length} resultados`);
      return results;
    } catch (error) {
      this.logger.error(`❌ Error en predicción: ${error.message}`);
      throw error;
    }
  }

  /**
   * Re-entrena el modelo con nuevos datos
   */
  async retrainModel(
    texts: string[],
    labels: number[],
  ): Promise<RetrainResponseDto> {
    try {
      // 1. Preprocesar textos (solo limpieza básica)
      this.logger.log(`📝 Preprocesando ${texts.length} textos para re-entrenamiento...`);
      const processedTexts = this.pipelineService.preprocessBatch(texts);

      // 2. Ejecutar re-entrenamiento
      this.logger.log('🔄 Ejecutando re-entrenamiento del modelo...');
      const metrics = await this.pythonBridge.retrain(processedTexts, labels);

      // 3. Formatear respuesta
      const response: RetrainResponseDto = {
        precision: metrics.precision,
        recall: metrics.recall,
        f1_score: metrics.f1_score,
        precision_per_class: metrics.precision_per_class,
        recall_per_class: metrics.recall_per_class,
        f1_score_per_class: metrics.f1_score_per_class,
        retrained_at: new Date().toISOString(),
        model_version: metrics.model_version,
        training_size: texts.length,
      };

      this.logger.log(`✅ Re-entrenamiento completado. F1-Score: ${metrics.f1_score}`);
      return response;
    } catch (error) {
      this.logger.error(`❌ Error en re-entrenamiento: ${error.message}`);
      throw error;
    }
  }
}
