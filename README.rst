Stream Likelihoods with ML
##########################

This is the PyTorch implementation of the StreamMapper code, which can be used to model stellar streams.
StreamMapper-PyTorch is a PyTorch framework for building Bayesian Mixture Density Networks, which can
then be trained using the standard PyTorch tooling.
Detailed explanations can be found in our paper (https://ui.adsabs.harvard.edu/abs/2023arXiv231116960S/abstract)
and especially in the code repository for the paper (https://github.com/nstarman/stellar_stream_density_ml_paper).

As an illustruative example:

.. code-block:: python

  bkg_phi2_model = sml.builtin.Uniform(
      data_scaler=scaler,
      indep_coord_names=("phi1",),
      coord_names=("phi2",),
      coord_bounds={"phi2": (lower, upper)},
      params=ModelParameters(),
  )

  bkg_plx_model = sml.builtin.Exponential(
      net=sml.nn.sequential(
          data=1, hidden_features=32, layers=3, features=1, dropout=0.15
      ),
      data_scaler=scaler,
      indep_coord_names=("phi1",),
      coord_names=("parallax",),
      coord_bounds={"parallax": (lower, upper)},
      params=ModelParameters(
          {"parallax": {"slope": ModelParameter(bounds=SigmoidBounds(15.0, 25.0))}}
      ),
  )


  bkg_flow = sml.builtin.compat.ZukoFlowModel(
      net=zuko.flows.MAF(features=2, context=1, transforms=4, hidden_features=[4] * 4),
      jacobian_logdet=-xp.log(xp.prod(...)),
      data_scaler=scaler[("phi1", "g", "r")],
      coord_names=phot_names,
      coord_bounds=phot_bounds,
      params=ModelParameters(),
  )

  background_model = sml.IndependentModels(
      {
          "astrometric": sml.IndependentModels(
              {"phi2": bkg_phi2_model, "parallax": bkg_plx_model}
          ),
          "photometric": bkg_flow,
      }
  )


  stream_astrometric_model = sml.builtin.Normal(
      net=...,  # PyTorch NN
      data_scaler=scaler,
      coord_names=coord_astrometric_names,
      coord_bounds=coord_astrometric_bounds,
      params=ModelParameters(
          {
              "phi2": {
                  "mu": ModelParameter(bounds=..., scaler=...),
                  "ln-sigma": ModelParameter(bounds=..., scaler=...),
              },
              "parallax": {
                  "mu": ModelParameter(bounds=..., scaler=...),
                  "ln-sigma": ModelParameter(bounds=..., scaler=...),
              },
          }
      ),
  )

  stream_isochrone_model = sml.builtin.IsochroneMVNorm(...)

  stream_model = sml.IndependentModels(
      {"astrometric": stream_astrometric_model, "photometric": stream_isochrone_model},
      unpack_params_hooks=(
          Parallax2DistMod(
              astrometric_coord="astrometric.parallax",
              photometric_coord="photometric.distmod",
          ),
      ),
  )

  model = sml.MixtureModel(
      {"stream": stream_model, "background": background_model},
      net=...,
      data_scaler=scaler,
      params=ModelParameters(
          {
              f"stream.ln-weight": ModelParameter(...),
              f"background.ln-weight": ModelParameter(...),
          }
      ),
  )
