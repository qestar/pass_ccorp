data = dict(
    train=dict(
        ds_dict=dict(
            type='CIFAR100_boxes',
            root='df',
            train=True,
        ),
        rcrop_dict=dict(
            type='cifar_train_rcrop',
            mean=1, std=1
        ),
        ccrop_dict=dict(
            type='cifar_train_ccrop',
            alpha=0.1,
            mean=1, std=1
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='CIFAR100',
            root=1,
            train=True,
        ),
        trans_dict=dict(
            type='cifar_test',
            mean=1, std=1
        ),
    )
)
print(data['train']['ccrop_dict'])