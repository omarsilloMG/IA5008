import optimization_methods
import sklearn.model_selection
import tqdm
import joblib
import pandas
import matplotlib.pyplot
import seaborn
import numpy

def plot_grid_search_figure():
    joblib.load("_grid_search_plot.pkl")

def read_learning_mins_info():
    return joblib.load("_learning_mins_info.pkl")

def plot_min_cost_heatmap(mins_info, l1_range, l2_range):
    heatmap = numpy.zeros((5,5))
    for l1 in mins_info.l1.unique():
        for l2 in mins_info.l2.unique():
            heatmap[int(l1)-2,int(l2)-2] = mins_info[
                (mins_info.l1 == l1) & (mins_info.l2 == l2) 
            ].min_value.iloc[0]

    import seaborn

    plot = seaborn.heatmap(
        heatmap,
        annot=True,
        cbar_kws={"label": "minimum average cost"}
    )

    plot.set_xlabel("l2")
    plot.set_xticklabels(range(*l2_range))
    plot.set_ylabel("l1")
    plot.set_yticklabels(range(*l1_range))

def plot_costs_and_accuracies(plot_title, training_costs, testing_costs, training_accuracies, testing_accuracies):
    fig, axs = matplotlib.pyplot.subplots(1,2,figsize=(10,3))

    collector = list()
    costs = {
        "training": training_costs,
        "testing": testing_costs
    }
    for name in costs:
        for epoch in range(costs[name].shape[1]):
            for fold, cost in enumerate(costs[name][:,epoch]):
                collector.append([
                    name, epoch, fold, cost
                ])
    collector = pandas.DataFrame(collector,columns="set epoch fold cost".split())
    
    seaborn.lineplot(
        collector,
        x="epoch",
        y="cost",
        hue="set",
        ax=axs[0]
    )
    mean = testing_costs.mean(axis=0)
    mean_min = mean.min()
    axs[0].grid()
    axs[0].set_title("Cost")
    axs[0].set_xlabel("Epoch (x10)")
    axs[0].set_ylabel(None)
    axs[0].axhline(mean_min, linestyle=":", color="gray", alpha=0.5)
    axs[0].axvline(numpy.where(mean == mean_min)[0][0], linestyle=":", color="gray", alpha=0.5)

    collector = list()
    accuracies = {
        "training": training_accuracies,
        "testing": testing_accuracies
    }
    for name in accuracies:
        for epoch in range(accuracies[name].shape[1]):
            for fold, accuracy in enumerate(accuracies[name][:,epoch]):
                collector.append([
                    name, epoch, fold, accuracy
                ])
    collector = pandas.DataFrame(collector,columns="set epoch fold accuracy".split())
    
    seaborn.lineplot(
        collector,
        x="epoch",
        y="accuracy",
        hue="set",
        ax=axs[1]
    )
    axs[1].grid()
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch (x10)")
    axs[1].set_ylabel(None)
    axs[1].axhline(testing_accuracies.mean(axis=0).max(), linestyle=":", color="gray", alpha=0.5)

    h, l = axs[1].get_legend_handles_labels()
    for ax in axs:
        ax.get_legend().remove()
    fig.legend(h, l, bbox_to_anchor=(1,1), loc="upper left" )

    fig.suptitle(plot_title)

    fig.tight_layout()

def run_and_get_model_costs_and_accuracies(images, labels, splitter, layers_dims, optimizer, learning_rate, num_epochs, decay=None, decay_rate=None):
    training_costs = list()
    testing_costs = list()
    training_accuracies = list()
    testing_accuracies = list()

    fold = -1
    for train_idxs, test_idxs in tqdm.tqdm(
        splitter.split(images.T, labels.squeeze()), 
        total=splitter.get_n_splits()
    ):
        fold += 1
        
        train_X = images[:,train_idxs]
        train_Y = labels[:,train_idxs]
        test_X = images[:,test_idxs]
        test_Y = labels[:,test_idxs]

        (
            parameters,
            training_cost,
            testing_cost,
            training_accuracy,
            testing_accuracy 
        ) = optimization_methods.model_for_learning_curve(
            train_X,
            train_Y,
            test_X,
            test_Y,
            layers_dims,
            optimizer = optimizer,
            learning_rate = learning_rate,
            mini_batch_size = train_X.shape[0],
            num_epochs = num_epochs,
            print_cost = False,
            decay=decay,
            decay_rate=decay_rate
        )

        training_costs.append(training_cost)
        testing_costs.append(testing_cost)
        training_accuracies.append(training_accuracy)
        testing_accuracies.append(testing_accuracy)

    return (
        numpy.vstack(training_costs),
        numpy.vstack(testing_costs),
        numpy.vstack(training_accuracies),
        numpy.vstack(testing_accuracies)
    )



def grid_search(images, labels, l1_range, l2_range, num_epochs):
    splitter = sklearn.model_selection.StratifiedKFold(5, shuffle=True, random_state=0)

    for l1 in range(*l1_range):
        for l2 in range(*l2_range):
            print(f"{l1=} {l2=}")
            layers_dims = (images.shape[0], l1, l2, 1)

            learning = list()
            fold = -1
            for train_idxs, test_idxs in tqdm.tqdm(
                splitter.split(images.T, labels.squeeze()), 
                total=splitter.get_n_splits()
            ):
                fold += 1
                
                train_X = images[:,train_idxs]
                train_Y = labels[:,train_idxs]
                test_X = images[:,test_idxs]
                test_Y = labels[:,test_idxs]

                parameters, costs, val_costs = optimization_methods.model_for_preliminary_experiment(
                    train_X,
                    train_Y,
                    test_X,
                    test_Y,
                    layers_dims,
                    optimizer = "gd",
                    learning_rate = 0.01,
                    mini_batch_size=train_X.shape[0],
                    num_epochs = num_epochs,
                    print_cost = False
                )

                for i, (cost, val_cost) in enumerate(zip(costs,val_costs)):
                    epoch = i*10
                    learning.append([ fold, epoch, "training", cost ])
                    learning.append([ fold, epoch, "testing", val_cost ])

            joblib.dump(
                pandas.DataFrame(learning, columns="fold epoch set cost".split()),
                f"_temp_{l1}_{l2}.pkl"
            )

def save_plot(l1_range, l2_range):
    num_of_l1s = l1_range[1]-l1_range[0]
    num_of_l2s = l2_range[1]-l2_range[0]

    fig, axs = matplotlib.pyplot.subplots(
        num_of_l1s, num_of_l2s,
        figsize=(3*num_of_l1s,3*num_of_l2s), sharex=True, sharey=True
    )
    axs = axs.flatten()

    idx = -1
    for l1 in tqdm.tqdm(range(*l1_range)):
        for l2 in range(*l2_range):
            idx += 1
            plot = seaborn.lineplot(
                joblib.load(f"_temp_{l1}_{l2}.pkl"),
                x="epoch",
                y="cost",
                hue="set",
                ax=axs[idx]
            )
            plot.set_title(f"{l1=}, {l2=}")
            plot.set_xlabel(None)
            plot.set_ylabel(None)
            plot.set_ylim(-0.04, 0.84)
            plot.grid()

    handles, labels = axs[idx].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
    for ax in axs:
        ax.legend().remove()

    fig.supxlabel("epoch")
    fig.supylabel("cost")
    fig.tight_layout()

    joblib.dump(fig,"_grid_search_plot.pkl")
    matplotlib.pyplot.close(fig)

def generate_learning_mins_info(l1_range, l2_range):
    collector = list()

    df = list()
    idx = -1
    for l1 in range(*l1_range):
        for l2 in range(*l2_range):
            idx += 1
            table = joblib.load(f"_temp_{l1}_{l2}.pkl")
            table["model"] = f"{l1}{l2}"
            df.append( table )
    df = pandas.concat(
        df,
        axis=0,
        ignore_index=True
    )
    df = df[ df.set == "testing" ]

    for _, mtable in df.groupby("model"):
        
        c = [ ftable["cost"].values for _, ftable in mtable.groupby("fold") ]

        cost_avg = numpy.vstack(c).mean(axis=0)

        collector.append(
            {
                "l1": mtable["model"].iloc[0][0],
                "l2": mtable["model"].iloc[0][1],
                "min_value": cost_avg.min(),
                "min_epoch": numpy.where( cost_avg == cost_avg.min() )[0]*10
            }
        )

    joblib.dump(
        pandas.DataFrame(collector),
        "_learning_mins_info.pkl"
    )

