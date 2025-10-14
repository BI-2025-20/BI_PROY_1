// ============================================
// src/app.module.ts
// ============================================

import { Module } from '@nestjs/common';
import { MLController } from './controllers/ml.controller';
import { MLService } from './services/ml.service';
import { PipelineService } from './services/pipeline.service';
import { PythonBridgeService } from './services/python-bridge.service';

@Module({
  imports: [],
  controllers: [MLController],
  providers: [MLService, PipelineService, PythonBridgeService],
})
export class AppModule {}
