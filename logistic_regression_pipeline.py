from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogisticRegressionPipeline:
    """A pipeline for training and evaluating a logistic regression model with optional label encoding.
    
    Attributes:
        model (Pipeline): The scikit-learn pipeline containing the scaler and logistic regression model.
        label_encoder (LabelEncoder, optional): The label encoder for encoding target labels.   
    """
    
    def __init__(self, label_encoder=None):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ])
        self.label_encoder = label_encoder

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def predict_composer_names(self, X_test):
        pred_composer_class = self.predict(X_test)
        if self.label_encoder is None:
            return pred_composer_class
        return self.label_encoder.inverse_transform(pred_composer_class)

    def validation_classification_report(self, X_val, y_val):
        from sklearn.metrics import classification_report, accuracy_score

        y_pred = self.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
            report = classification_report(
                y_val,
                y_pred,
                target_names=target_names,
                digits=4,
            )
            pred_names = self.label_encoder.inverse_transform(y_pred)
        else:
            report = classification_report(y_val, y_pred, digits=4)
            pred_names = y_pred

        return {
            "accuracy": accuracy,
            "report": report,
            "y_pred": y_pred,
            "y_pred_names": pred_names,
        }
