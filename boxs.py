def train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_iter = len(train_loader)
    end = time.time()
    time1 = time.time()
    for idx, (images, _) in enumerate(train_loader):
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # measure data time
        data_time.update(time.time() - end)

        # compute output
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -0.5 * (criterion(p1, z2).mean() + criterion(p2, z1).mean())
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {loss:.3f}({losses.avg:.3f})')

    if logger is not None:  # cfg.rank == 0
        time2 = time.time()
        epoch_time = format_time(time2 - time1)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)