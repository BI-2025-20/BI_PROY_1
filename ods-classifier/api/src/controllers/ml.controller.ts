// ============================================
// src/controllers/ml.controller.ts
// ============================================

import {
  Controller,
  Post,
  Body,
  HttpException,
  HttpStatus,
  UsePipes,
  ValidationPipe,
  Logger,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { PredictionResultDto, PredictRequestDto } from 'src/dto/prediction.dto';
import { RetrainRequestDto, RetrainResponseDto } from 'src/dto/retrain.dto';
import { MLService } from 'src/services/ml.service';

@ApiTags('Machine Learning')
@Controller('api/ml')
@UsePipes(new ValidationPipe({ transform: true, whitelist: true }))
export class MLController {
  private readonly logger = new Logger(MLController.name);

  constructor(private readonly mlService: MLService) {}

  @Post('predict')
  @ApiOperation({ 
    summary: 'Predice ODS para textos dados',
    description: 'Recibe uno o más textos y retorna las predicciones de ODS con sus probabilidades'
  })
  @ApiResponse({ status: 200, description: 'Predicciones exitosas', type: [PredictionResultDto] })
  @ApiResponse({ status: 400, description: 'Datos de entrada inválidos' })
  @ApiResponse({ status: 500, description: 'Error interno del servidor' })
  async predict(
    @Body() body: PredictRequestDto,
  ): Promise<PredictionResultDto[]> {
    try {
      this.logger.log(`Recibida solicitud de predicción para ${body.instances.length} instancias`);
      
      const texts = body.instances.map(inst => inst.textos);
      const predictions = await this.mlService.predictBatch(texts);
      
      this.logger.log(`Predicciones completadas exitosamente`);
      return predictions;
    } catch (error) {
      this.logger.error(`Error en endpoint /predict: ${error.message}`);
      throw new HttpException(
        {
          statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
          message: 'Error al realizar predicciones',
          error: error.message,
        },
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  @Post('retrain')
  @ApiOperation({ 
    summary: 'Re-entrena el modelo con nuevos datos',
    description: 'Recibe instancias etiquetadas y re-entrena el modelo, retornando métricas de desempeño'
  })
  @ApiResponse({ status: 200, description: 'Re-entrenamiento exitoso', type: RetrainResponseDto })
  @ApiResponse({ status: 400, description: 'Datos de entrada inválidos' })
  @ApiResponse({ status: 500, description: 'Error interno del servidor' })
  async retrain(
    @Body() body: RetrainRequestDto,
  ): Promise<RetrainResponseDto> {
    try {
      this.logger.log(`Recibida solicitud de re-entrenamiento con ${body.instances.length} instancias`);
      
      const texts = body.instances.map(inst => inst.textos);
      const labels = body.instances.map(inst => inst.labels);
      
      // Validar que solo haya labels válidos (1, 3, 4)
      const validLabels = new Set([1, 3, 4]);
      const invalidLabels = labels.filter(l => !validLabels.has(l));
      if (invalidLabels.length > 0) {
        throw new HttpException(
          `Labels inválidos encontrados: ${invalidLabels}. Solo se permiten 1, 3, 4`,
          HttpStatus.BAD_REQUEST,
        );
      }
      
      const metrics = await this.mlService.retrainModel(texts, labels);
      
      this.logger.log(`Re-entrenamiento completado. F1-Score: ${metrics.f1_score}`);
      return metrics;
    } catch (error) {
      this.logger.error(`Error en endpoint /retrain: ${error.message}`);
      throw new HttpException(
        {
          statusCode: error.status || HttpStatus.INTERNAL_SERVER_ERROR,
          message: 'Error al re-entrenar modelo',
          error: error.message,
        },
        error.status || HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}