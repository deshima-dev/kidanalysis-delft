import mkid_data as md

TemperatureData = md.named_array('TemperatureData', ['temperature'], ['Temperature [K]'])
PowerData = md.named_array('PowerData', ['power'], ['Power [W]'])

class TemperatureTODData(md.BaseMultiData, md.TimeArray, TemperatureData):
    """
    a class for PSD data after fitting: FFTFreqArray + KidFitResponse.

    - **FixedFitPSDData(frequency, index, I, Q, fsample)**:

    :param time: 1-D array of time
    :param temperature: 1-D array of temperature [K]
    """
    def __init__(self, time, temperature):
        xdata = md.TimeArray(time)
        ydata = TemperatureData((temperature,))
        super(TemperatureTODData, self).__init__((xdata, ydata))

class PowerTODData(md.BaseMultiData, md.TimeArray, PowerData):
    """
    a class for PSD data after fitting: FFTFreqArray + KidFitResponse.

    - **FixedFitPSDData(frequency, index, I, Q, fsample)**:

    :param time: 1-D array of time
    :param power: 1-D array of power [W]
    """
    def __init__(self, time, power):
        xdata = md.TimeArray(time)
        ydata = TemperatureData((power,))
        super(PowerTODData, self).__init__((xdata, ydata))

AveragedPSDData = md.named_array('AveragedPSDData',
                                 ['amplitude', 'phase', 'damplitude', 'dphase'])


class PowerAveragedPSDData(md.BaseMultiData, PowerData, AveragedPSDData):
    """
    a class for PSD data after fitting: FFTFreqArray + KidFitResponse.

    - **FixedFitPSDData(frequency, index, I, Q, fsample)**:

    :param power: 1-D array of power [W]
    :param amplitude: 1-D array of amplitude
    :param phase: 1-D array of phase
    :param damplitude: 1-D array of amplitude error
    :param dphase: 1-D array of phase error
    """
    def __init__(self, power, amplitude, phase, damplitude, dphase):
        xdata = PowerData(power)
        ydata = AveragedPSDData((amplitude, phase, damplitude, dphase))
        super(PowerAveragedPSDData, self).__init__((xdata, ydata))

ResponsivityData = md.named_array('ResponsivityData',
                                  ['power', 'amplitude', 'phase',
                                   'dpower', 'damplitude', 'dphase'],
                                  ['P_rad [W]',
                                   'dA/d_Prad [/W]', 'dPhase/dP_rad [/W]',
                                   'delta(power) [W]',
                                   'delta(dA/d_Prad) [/W]', 'delta(dPhase/dP_rad) [/W]'])
