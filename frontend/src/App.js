import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
    const featureNames = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
        'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error', 'concave points error',
        'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture',
        'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
        'worst concavity', 'worst concave points', 'worst symmetry',
        'worst fractal dimension'
    ];

    const [features, setFeatures] = useState(() =>
        featureNames.reduce((acc, name) => ({ ...acc, [name]: '' }), {})
    );
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showCorrelations, setShowCorrelations] = useState(false);
    const [meanCorrelationImage, setMeanCorrelationImage] = useState(null);
    const [worstCorrelationImage, setWorstCorrelationImage] = useState(null);
    const [errorCorrelationImage, setErrorCorrelationImage] = useState(null);
    const [globalCorrelationImage, setGlobalCorrelationImage] = useState(null);
    const [loadingCorrelations, setLoadingCorrelations] = useState(true);

    const handleShowCorrelations = () => setShowCorrelations(true);
    const handleShowPrediction = () => setShowCorrelations(false);

    useEffect(() => {
        const fetchImages = async () => {
            try {
                const responseGlobal = await fetch('/correlation_global');
                const blobGlobal = await responseGlobal.blob();
                setGlobalCorrelationImage(URL.createObjectURL(blobGlobal));

                const responseMean = await fetch('/correlation_mean');
                const blobMean = await responseMean.blob();
                setMeanCorrelationImage(URL.createObjectURL(blobMean));

                const responseWorst = await fetch('/correlation_worst');
                const blobWorst = await responseWorst.blob();
                setWorstCorrelationImage(URL.createObjectURL(blobWorst));

                const responseError = await fetch('/correlation_error');
                const blobError = await responseError.blob();
                setErrorCorrelationImage(URL.createObjectURL(blobError));
            } catch (error) {
                console.error("Erreur lors du chargement des matrices de corrélation:", error);
            } finally {
                setLoadingCorrelations(false);
            }
        };

        if (showCorrelations) {
            fetchImages();
        }
    }, [showCorrelations]);

    const handleChange = (event) => {
        const { name, value } = event.target;
        const numericValue = parseFloat(value);
        setFeatures(prev => ({
            ...prev,
            [name]: isNaN(numericValue) ? '' : numericValue,
        }));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(features),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erreur lors de la prédiction');
            }

            const data = await response.json();
            setPrediction(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="App">
            <h1>Breast Cancer Prediction</h1>
            <button onClick={handleShowPrediction}>Show Prediction</button>
            <button onClick={handleShowCorrelations}>Show Correlations</button>

            {!showCorrelations && (
                <form onSubmit={handleSubmit}>
                    {featureNames.map(name => (
                        <div key={name}>
                            <label htmlFor={name}>{name.replace(/_/g, ' ').replace('mean', 'Mean').replace('se', 'Erreur Standard').replace('worst', 'Worst')} :</label>
                            <input
                                type="number"
                                id={name}
                                name={name}
                                value={features[name]}
                                onChange={handleChange}
                                placeholder={`Entrez la valeur pour ${name}`}
                                required
                            />
                        </div>
                    ))}
                    <button type="submit">Predict</button>
                    {loading && <p>Chargement...</p>}
                    {error && <p className="error">Erreur: {error}</p>}
                    {prediction && (
                        <div className="prediction-result">
                            <h3>Prediction result</h3>
                            <p>Probability of Malignancy: {(prediction.probability * 100).toFixed(2)}%</p>
                            <p>Diagnosis: {prediction.diagnosis}</p>
                        </div>
                    )}
                </form>
            )}

            {showCorrelations && (
                <div>
                    <h2>Correlation Matrices of the Features</h2>
                    {loadingCorrelations && <p>Chargement des matrices de corrélation...</p>}

                    <div style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                        <div style={{ margin: '400px', flex: '1 1 50%', minWidth: '300px' }}>
                            <h3>Correlation  - Global</h3>
                            {globalCorrelationImage && <img src={globalCorrelationImage} alt="Matrice de Corrélation - Globale" style={{ maxWidth: '100%', height: 'auto' }} />}
                            {!globalCorrelationImage && <p>Chargement de la matrice de corrélation (Global)...</p>}
                        </div>

                        <div style={{ margin: '400px', flex: '1 1 50%', minWidth: '300px' }}>
                            <h3>Correlation  - Mean features</h3>
                            {meanCorrelationImage && <img src={meanCorrelationImage} alt="Matrice de Corrélation - Mean" style={{ maxWidth: '100%', height: 'auto' }} />}
                            {!meanCorrelationImage && <p>Erreur lors du chargement de la matrice de corrélation (Mean).</p>}
                        </div>

                        <div style={{ margin: '400px', flex: '1 1 50%', minWidth: '300px' }}>
                            <h3>Correlation  - Worst features</h3>
                            {worstCorrelationImage && <img src={worstCorrelationImage} alt="Matrice de Corrélation - Worst" style={{ maxWidth: '100%', height: 'auto' }} />}
                            {!worstCorrelationImage && <p>Erreur lors du chargement de la matrice de corrélation (Worst).</p>}
                        </div>

                        <div style={{ margin: '400px', flex: '1 1 50%', minWidth: '300px' }}>
                            <h3>Correlation -Error features</h3>
                            {errorCorrelationImage && <img src={errorCorrelationImage} alt="Matrice de Corrélation - Error" style={{ maxWidth: '100%', height: 'auto' }} />}
                            {!errorCorrelationImage && <p>Erreur lors du chargement de la matrice de corrélation (Error).</p>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
