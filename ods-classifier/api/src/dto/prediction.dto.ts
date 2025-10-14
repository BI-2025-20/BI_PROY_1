// ============================================
// 1. DTOs (Data Transfer Objects)
// src/dto/prediction.dto.ts
// ============================================

import { IsString, IsNotEmpty, IsArray, ValidateNested, IsNumber, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export class TextInstanceDto {
  @ApiProperty({ 
    description: 'Texto de la opinión ciudadana a clasificar',
    example: 'La educación digital es fundamental para el desarrollo de competencias del siglo XXI'
  })
  @IsString()
  @IsNotEmpty({ message: 'El campo textos no puede estar vacío' })
  textos: string;
}

export class PredictRequestDto {
  @ApiProperty({ 
    type: [TextInstanceDto],
    description: 'Array de instancias de texto para predecir'
  })
  @IsArray()
  @ValidateNested({ each: true })
  @Type(() => TextInstanceDto)
  instances: TextInstanceDto[];
}

export class PredictionResultDto {
  @ApiProperty({ description: 'ODS predicho (1, 3 o 4)' })
  predicted_label: number;

  @ApiProperty({ description: 'Probabilidad de la predicción' })
  probability: number;

  @ApiProperty({ description: 'Texto original' })
  textos: string;

  @ApiProperty({ description: 'Distribución de probabilidades por clase' })
  class_probabilities: { [key: string]: number };
}