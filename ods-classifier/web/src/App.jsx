import React, { useState } from 'react';
import { Send, RefreshCw, Users, GraduationCap, Heart, TrendingUp, AlertCircle, CheckCircle, FileText, BarChart3 } from 'lucide-react';

const ODSApp = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [inputText, setInputText] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Estado para re-entrenamiento
  const [retrainData, setRetrainData] = useState([{ textos: '', labels: 1 }]);
  const [retrainMetrics, setRetrainMetrics] = useState(null);
  const [retrainLoading, setRetrainLoading] = useState(false);

  const API_URL = 'http://localhost:3001/api/ml';

  const odsInfo = {
    1: {
      title: 'ODS 1: Fin de la Pobreza',
      icon: <Users className="w-6 h-6" />,
      color: 'bg-red-500',
      description: 'Erradicar la pobreza en todas sus formas'
    },
    3: {
      title: 'ODS 3: Salud y Bienestar',
      icon: <Heart className="w-6 h-6" />,
      color: 'bg-green-500',
      description: 'Garantizar vidas saludables y bienestar'
    },
    4: {
      title: 'ODS 4: Educación de Calidad',
      icon: <GraduationCap className="w-6 h-6" />,
      color: 'bg-blue-500',
      description: 'Educación inclusiva y equitativa de calidad'
    }
  };

  const handlePredict = async () => {
    if (!inputText.trim()) {
      setError('Por favor ingrese un texto');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instances: [{ textos: inputText }]
        })
      });

      if (!response.ok) throw new Error('Error en la predicción');

      const results = await response.json();
      setPredictions(results);
    } catch (err) {
      setError('Error al conectar con la API: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    // Validar datos
    const validData = retrainData.filter(item => item.textos.trim() !== '');
    
    if (validData.length < 5) {
      setError('Se requieren al menos 5 instancias para re-entrenar');
      return;
    }

    setRetrainLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instances: validData })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Error en el re-entrenamiento');
      }

      const metrics = await response.json();
      setRetrainMetrics(metrics);
      setError('');
    } catch (err) {
      setError('Error al re-entrenar: ' + err.message);
    } finally {
      setRetrainLoading(false);
    }
  };

  const addRetrainRow = () => {
    setRetrainData([...retrainData, { textos: '', labels: 1 }]);
  };

  const updateRetrainRow = (index, field, value) => {
    const newData = [...retrainData];
    newData[index][field] = field === 'labels' ? parseInt(value) : value;
    setRetrainData(newData);
  };

  const removeRetrainRow = (index) => {
    const newData = retrainData.filter((_, i) => i !== index);
    setRetrainData(newData.length > 0 ? newData : [{ textos: '', labels: 1 }]);
  };

  const PredictionCard = ({ prediction }) => {
    const ods = odsInfo[prediction.predicted_label];
    const probability = (prediction.probability * 100).toFixed(1);

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-4 border-l-4" style={{ borderLeftColor: ods.color.replace('bg-', '#') }}>
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`${ods.color} text-white p-3 rounded-lg`}>
              {ods.icon}
            </div>
            <div>
              <h3 className="font-bold text-lg">{ods.title}</h3>
              <p className="text-sm text-gray-600">{ods.description}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-800">{probability}%</div>
            <div className="text-xs text-gray-500">Confianza</div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg mb-4">
          <p className="text-sm text-gray-700 italic">"{prediction.textos}"</p>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-semibold text-gray-700 mb-2">Distribución de Probabilidades:</div>
          {Object.entries(prediction.class_probabilities).map(([key, value]) => {
            const odsNum = parseInt(key.split('_')[1]);
            const percentage = (value * 100).toFixed(1);
            return (
              <div key={key} className="flex items-center gap-2">
                <div className="w-24 text-sm text-gray-600">{odsInfo[odsNum].title.split(':')[0]}</div>
                <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full ${odsInfo[odsNum].color} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div className="w-12 text-sm text-gray-700 font-medium">{percentage}%</div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                Clasificador de ODS - UNFPA
              </h1>
              <p className="text-gray-600">
                Sistema de análisis automático de opiniones ciudadanas para Objetivos de Desarrollo Sostenible
              </p>
            </div>
            <div className="flex gap-2">
              <div className={`w-3 h-3 rounded-full ${odsInfo[1].color}`} />
              <div className={`w-3 h-3 rounded-full ${odsInfo[3].color}`} />
              <div className={`w-3 h-3 rounded-full ${odsInfo[4].color}`} />
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="flex border-b">
            <button
              onClick={() => setActiveTab('predict')}
              className={`flex-1 py-4 px-6 font-medium transition-colors ${
                activeTab === 'predict'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <FileText className="w-5 h-5" />
                Análisis de Texto
              </div>
            </button>
            <button
              onClick={() => setActiveTab('retrain')}
              className={`flex-1 py-4 px-6 font-medium transition-colors ${
                activeTab === 'retrain'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-600 hover:bg-gray-50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <RefreshCw className="w-5 h-5" />
                Re-entrenamiento (Experto)
              </div>
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Prediction Tab */}
        {activeTab === 'predict' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Analizar Opinión Ciudadana</h2>
              
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ingrese aquí la opinión ciudadana que desea clasificar según los ODS..."
                className="w-full p-4 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none min-h-32 mb-4"
              />

              <button
                onClick={handlePredict}
                disabled={loading || !inputText.trim()}
                className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
                    Analizando...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Clasificar Texto
                  </>
                )}
              </button>
            </div>

            {/* Predictions Results */}
            {predictions.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <CheckCircle className="w-6 h-6 text-green-500" />
                  Resultados del Análisis
                </h2>
                {predictions.map((pred, idx) => (
                  <PredictionCard key={idx} prediction={pred} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Retrain Tab */}
        {activeTab === 'retrain' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-800">Panel de Re-entrenamiento</h2>
                <button
                  onClick={addRetrainRow}
                  className="bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg transition-colors"
                >
                  + Agregar Instancia
                </button>
              </div>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded">
                <p className="text-sm text-blue-800">
                  <strong>Nota:</strong> Se requieren al menos 5 instancias etiquetadas para re-entrenar el modelo.
                  El nuevo modelo reemplazará al actual y estará disponible inmediatamente para predicciones.
                </p>
              </div>

              <div className="space-y-4 mb-6 max-h-96 overflow-y-auto">
                {retrainData.map((item, idx) => (
                  <div key={idx} className="flex gap-3 items-start bg-gray-50 p-4 rounded-lg">
                    <div className="flex-1">
                      <textarea
                        value={item.textos}
                        onChange={(e) => updateRetrainRow(idx, 'textos', e.target.value)}
                        placeholder="Texto de la opinión..."
                        className="w-full p-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none min-h-20 text-sm"
                      />
                    </div>
                    <div className="w-40">
                      <select
                        value={item.labels}
                        onChange={(e) => updateRetrainRow(idx, 'labels', e.target.value)}
                        className="w-full p-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none text-sm"
                      >
                        <option value={1}>ODS 1: Pobreza</option>
                        <option value={3}>ODS 3: Salud</option>
                        <option value={4}>ODS 4: Educación</option>
                      </select>
                    </div>
                    {retrainData.length > 1 && (
                      <button
                        onClick={() => removeRetrainRow(idx)}
                        className="p-3 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>

              <button
                onClick={handleRetrain}
                disabled={retrainLoading}
                className="w-full bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {retrainLoading ? (
                  <>
                    <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin" />
                    Re-entrenando Modelo...
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-5 h-5" />
                    Re-entrenar Modelo
                  </>
                )}
              </button>
            </div>

            {/* Metrics Display */}
            {retrainMetrics && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-purple-500" />
                  Métricas del Modelo Re-entrenado
                </h2>

                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-blue-700">
                      {(retrainMetrics.precision * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-blue-600 font-medium mt-1">Precisión</div>
                  </div>
                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-green-700">
                      {(retrainMetrics.recall * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-green-600 font-medium mt-1">Recall</div>
                  </div>
                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-purple-700">
                      {(retrainMetrics.f1_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-purple-600 font-medium mt-1">F1-Score</div>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <h3 className="font-bold text-gray-700 mb-3">Métricas por Clase:</h3>
                  <div className="space-y-3">
                    {[1, 3, 4].map(ods => (
                      <div key={ods} className="bg-gray-50 p-3 rounded-lg">
                        <div className="font-medium text-gray-800 mb-2">{odsInfo[ods].title}</div>
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div>
                            <span className="text-gray-600">Precisión: </span>
                            <span className="font-semibold">
                              {(retrainMetrics.precision_per_class[`ODS_${ods}`] * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-600">Recall: </span>
                            <span className="font-semibold">
                              {(retrainMetrics.recall_per_class[`ODS_${ods}`] * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-600">F1: </span>
                            <span className="font-semibold">
                              {(retrainMetrics.f1_score_per_class[`ODS_${ods}`] * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-4 text-sm text-gray-600">
                  <p><strong>Versión:</strong> {retrainMetrics.model_version}</p>
                  <p><strong>Tamaño de entrenamiento:</strong> {retrainMetrics.training_size} instancias</p>
                  <p><strong>Fecha:</strong> {new Date(retrainMetrics.retrained_at).toLocaleString('es-ES')}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ODSApp;
