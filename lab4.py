"""
CS3081 Artificial Intelligence — Lab 4: Bayesian Network (Likelihood and Inference)
Group Members:
- Ali Ahmed (ID: S24109004)
- Abubaker Baobaid (ID: S23108655)
- Abdulrahman Al Shalhoub (ID: S24109270)
Instructor: Dr. Naila Marir
Effat University · College of Engineering
"""

import warnings
warnings.filterwarnings("ignore")

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_model():
    print("=" * 70)
    print("STEP 2: BUILDING THE BAYESIAN NETWORK")
    print("=" * 70)

    # 1) The graph: each pair is an arrow (parent, child)
    model = DiscreteBayesianNetwork([
        ("rain", "maintenance"),
        ("rain", "train"),
        ("maintenance", "train"),
        ("train", "appointment"),
    ])

    # 2) P(Rain): no parents, so one column
    rain = TabularCPD(
        variable="rain",
        variable_card=3,
        values=[[0.7],  # none
                [0.2],  # light
                [0.1]], # heavy
        state_names={"rain": ["none", "light", "heavy"]}
    )

    # 3) P(Maintenance | Rain): one column for each value of rain
    #                      rain = none  light  heavy
    maintenance = TabularCPD(
        variable="maintenance",
        variable_card=2,
        values=[[0.4, 0.2, 0.1],  # yes
                [0.6, 0.8, 0.9]], # no
        evidence=["rain"],
        evidence_card=[3],
        state_names={"maintenance": ["yes", "no"],
                     "rain": ["none", "light", "heavy"]}
    )

    # 4) P(Train | Rain, Maintenance): one column for each (rain, maintenance) pair
    #                           (none,yes) (none,no) (light,yes) (light,no) (heavy,yes) (heavy,no)
    train = TabularCPD(
        variable="train",
        variable_card=2,
        values=[[0.8, 0.9, 0.6, 0.7, 0.4, 0.5],  # on time
                [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]], # delayed
        evidence=["rain", "maintenance"],
        evidence_card=[3, 2],
        state_names={"train": ["on time", "delayed"],
                     "rain": ["none", "light", "heavy"],
                     "maintenance": ["yes", "no"]}
    )

    # 5) P(Appointment | Train): one column for each value of train
    #                                train = on time  delayed
    appointment = TabularCPD(
        variable="appointment",
        variable_card=2,
        values=[[0.9, 0.6],  # attend
                [0.1, 0.4]], # miss
        evidence=["train"],
        evidence_card=[2],
        state_names={"appointment": ["attend", "miss"],
                     "train": ["on time", "delayed"]}
    )

    # 6) Put tables into graph and check them
    model.add_cpds(rain, maintenance, train, appointment)
    is_valid = model.check_model()
    print("Model is correct:", is_valid)
    return model, rain, maintenance, train, appointment


def run_lab():
    model, rain, maintenance, train, appointment = build_model()

    print("\n" + "=" * 70)
    print("STEP 3: JOINT PROBABILITY (LIKELIHOOD) — SCENARIO 1: A GOOD DAY")
    print("P(rain=none, maintenance=no, train='on time', appointment=attend)")
    print("=" * 70)
    p3 = (rain.get_value(rain="none") *
          maintenance.get_value(maintenance="no", rain="none") *
          train.get_value(train="on time", rain="none", maintenance="no") *
          appointment.get_value(appointment="attend", train="on time"))
    print(f"Factors: P(none)={rain.get_value(rain='none')} * P(no|none)={maintenance.get_value(maintenance='no', rain='none')} * P(on time|none,no)={train.get_value(train='on time', rain='none', maintenance='no')} * P(attend|on time)={appointment.get_value(appointment='attend', train='on time')}")
    print(f"Joint Probability: {p3:.4f}")

    print("\n" + "=" * 70)
    print("STEP 4: JOINT PROBABILITY (LIKELIHOOD) — SCENARIO 2: A BAD DAY")
    print("P(rain=heavy, maintenance=yes, train=delayed, appointment=miss)")
    print("=" * 70)
    p4 = (rain.get_value(rain="heavy") *
          maintenance.get_value(maintenance="yes", rain="heavy") *
          train.get_value(train="delayed", rain="heavy", maintenance="yes") *
          appointment.get_value(appointment="miss", train="delayed"))
    print(f"Factors: P(heavy)={rain.get_value(rain='heavy')} * P(yes|heavy)={maintenance.get_value(maintenance='yes', rain='heavy')} * P(delayed|heavy,yes)={train.get_value(train='delayed', rain='heavy', maintenance='yes')} * P(miss|delayed)={appointment.get_value(appointment='miss', train='delayed')}")
    print(f"Joint Probability: {p4:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5: INFERENCE WITH EVIDENCE — TRAIN IS DELAYED")
    print("=" * 70)
    inference = VariableElimination(model)

    def show(evidence):
        """Print P(X | evidence) for every node X that we did not observe."""
        for node in ["rain", "maintenance", "train", "appointment"]:
            if node in evidence:
                print(f"{node}: {evidence[node]} (observed)")
                continue
            result = inference.query([node], evidence=evidence, show_progress=False)
            print(node)
            for value, prob in zip(result.state_names[node], result.values):
                print(f"  {value}: {prob:.4f}")

    show({"train": "delayed"})

    print("\n" + "=" * 70)
    print("STEP 6: MANUAL ENUMERATION INFERENCE (BY HAND)")
    print("P(rain | train=delayed) = alpha * Sum_m P(rain, m, train=delayed)")
    print("=" * 70)
    unnorm_none = 0.7 * (0.4 * 0.2 + 0.6 * 0.1)
    unnorm_light = 0.2 * (0.2 * 0.4 + 0.8 * 0.3)
    unnorm_heavy = 0.1 * (0.1 * 0.6 + 0.9 * 0.5)
    total_sum = unnorm_none + unnorm_light + unnorm_heavy
    norm_none = unnorm_none / total_sum
    norm_light = unnorm_light / total_sum
    norm_heavy = unnorm_heavy / total_sum

    print(f"Unnormalized P(none, delayed):  {unnorm_none:.3f}")
    print(f"Unnormalized P(light, delayed): {unnorm_light:.3f}")
    print(f"Unnormalized P(heavy, delayed): {unnorm_heavy:.3f}")
    print(f"Total Sum (Evidence Normalizer): {total_sum:.3f}")
    print(f"Normalized P(rain=none | delayed):  {norm_none:.4f}")
    print(f"Normalized P(rain=light | delayed): {norm_light:.4f}")
    print(f"Normalized P(rain=heavy | delayed): {norm_heavy:.4f}")

    print("\n" + "=" * 70)
    print("STEP 8: REASONING UPWARD (DIAGNOSTIC) — MISSED APPOINTMENT")
    print("=" * 70)
    show({"appointment": "miss"})

    print("\n" + "=" * 70)
    print("STEP 9: TWO OBSERVATIONS (INTERCAUSAL) — DELAYED TRAIN & HEAVY RAIN")
    print("=" * 70)
    show({"train": "delayed", "rain": "heavy"})

    maint_hand = (0.1 * 0.6) / (0.1 * 0.6 + 0.9 * 0.5)
    print(f"Step 9 Hand Calculation P(maint=yes | delayed, heavy): {maint_hand:.4f}")


if __name__ == "__main__":
    run_lab()
