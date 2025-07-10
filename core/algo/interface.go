package algo

import (
	"recommend-sys/data"
)

type Algorithm interface {
	Predict(userID, itemID int) float64
	Fit(trainData data.Set, options ...OptionsEditor)
}

type Option struct {
	reg           float64
	regUserFactor float64
	regItemFactor float64
	lr            float64
	lrUserFactor  float64
	lrItemFactor  float64
	biased        bool
	nEpoch        int
	nFactors      int
	initMean      float64
	initStdDev    float64
	initLow       float64
	initHigh      float64
}

type OptionsEditor func(option *Option)

func SetLearningRate(leaningRate float64) OptionsEditor {
	return func(option *Option) {
		option.lr = leaningRate
	}
}

func SetRegularization(regularization float64) OptionsEditor {
	return func(option *Option) {
		// 正则化
		option.reg = regularization
	}
}

func SetNEpoch(nEpoch int) OptionsEditor {
	return func(option *Option) {
		option.nEpoch = nEpoch
	}
}
func SetNFactors(nFactors int) OptionsEditor {
	return func(option *Option) {
		option.nFactors = nFactors
	}
}
