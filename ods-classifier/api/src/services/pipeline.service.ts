// ============================================
// 3. Pipeline Service
// src/services/pipeline.service.ts
// ============================================

import { Injectable, Logger } from '@nestjs/common';

@Injectable()
export class PipelineService {
  private readonly logger = new Logger(PipelineService.name);

  constructor() {
    this.logger.log('Pipeline Service iniciado - Modo: Limpieza básica únicamente');
  }


  preprocessText(text: string): string {
    try {
      // Solo limpieza básica:
      // 1. Convertir a minúsculas
      // 2. Normalizar espacios en blanco
      // 3. Eliminar saltos de línea múltiples
      // 4. Trim
      
      const cleanedText = text
        .toLowerCase()
        .replace(/\s+/g, ' ')           // Múltiples espacios → un espacio
        .replace(/\n+/g, ' ')           // Saltos de línea → espacio
        .trim();
      
      this.logger.debug(`Texto limpio (${text.length} → ${cleanedText.length} chars)`);
      
      return cleanedText;
    } catch (error) {
      this.logger.error(`Error en limpieza básica: ${error.message}`);
      throw error;
    }
  }

  /**
   * Procesa un lote de textos
   */
  preprocessBatch(texts: string[]): string[] {
    return texts.map(text => this.preprocessText(text));
  }

  /**
   * Validación de entrada
   */
  validateText(text: string): { isValid: boolean; error?: string } {
    if (!text || text.trim().length === 0) {
      return { isValid: false, error: 'El texto no puede estar vacío' };
    }

    if (text.trim().length < 10) {
      return { 
        isValid: false, 
        error: 'El texto es demasiado corto (mínimo 10 caracteres)' 
      };
    }

    if (text.length > 10000) {
      return { 
        isValid: false, 
        error: 'El texto es demasiado largo (máximo 10,000 caracteres)' 
      };
    }

    return { isValid: true };
  }
}

