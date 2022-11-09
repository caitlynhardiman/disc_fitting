import pymcfost as mcfost

def convolve(mcfost_model):
    model = []
    for i in range(len(mcfost_model.lines)):
        mcfost_model.plot_map(iv=i, bmaj=exocube.bmaj, bmin=exocube.bmin, bpa=exocube.bpa)
        model.append(mcfost_model.last_im[165:880, 180:855])
        plt.close()
    return model
