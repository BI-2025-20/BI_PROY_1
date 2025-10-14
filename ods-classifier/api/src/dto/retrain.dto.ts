import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsNotEmpty, IsArray, ValidateNested, IsNumber, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';
// ============================================
// 2. DTOs para Re-entrenamiento
// src/dto/reentrain.dto.ts
// ============================================

export class LabeledInstanceDto {
  @ApiProperty({ 
    description: 'Texto de la opinión ciudadana',
    example: 'La salud materno-infantil requiere mayor inversión'
  })
  @IsString()
  @IsNotEmpty()
  textos: string;

  @ApiProperty({ 
    description: 'Etiqueta ODS (1: Sin Pobreza, 3: Salud, 4: Educación)',
    example: 3,
    enum: [1, 3, 4]
  })
  @IsNumber()
  @Min(1)
  @Max(4)
  labels: number;
}

export class RetrainRequestDto {
  @ApiProperty({ 
    type: [LabeledInstanceDto],
    description: 'Array de instancias etiquetadas para re-entrenamiento'
  })
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => LabeledInstanceDto)
  instances: LabeledInstanceDto[];
}

export class RetrainResponseDto {
  @ApiProperty({ description: 'Precisión del modelo re-entrenado' })
  precision: number;

  @ApiProperty({ description: 'Recall del modelo re-entrenado' })
  recall: number;

  @ApiProperty({ description: 'F1-Score del modelo re-entrenado' })
  f1_score: number;

  @ApiProperty({ description: 'Precisión por clase' })
  precision_per_class: { [key: string]: number };

  @ApiProperty({ description: 'Recall por clase' })
  recall_per_class: { [key: string]: number };

  @ApiProperty({ description: 'F1-Score por clase' })
  f1_score_per_class: { [key: string]: number };

  @ApiProperty({ description: 'Timestamp del re-entrenamiento' })
  retrained_at: string;

  @ApiProperty({ description: 'Versión del modelo' })
  model_version: string;

  @ApiProperty({ description: 'Número de instancias usadas' })
  training_size: number;
}