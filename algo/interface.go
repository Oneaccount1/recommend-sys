package algo

import "recommend-sys/data"

type Algorithm interface {
	Predict(userID, itemID int) float64
	Fit(trainData data.Set, options ...OptionsEditor)
}

type Option struct {
	regularization float64
	learningRate   float64
	nEpoch         int
}

type OptionsEditor func(option *Option)

func SetLearningRate(leaningRate float64) OptionsEditor {
	return func(option *Option) {
		option.learningRate = leaningRate
	}
}

func SetRegularization(regularization float64) OptionsEditor {
	return func(option *Option) {
		option.regularization = regularization
	}
}

func SetNEpoch(nEpoch int) OptionsEditor {
	return func(option *Option) {
		option.nEpoch = nEpoch
	}
}
