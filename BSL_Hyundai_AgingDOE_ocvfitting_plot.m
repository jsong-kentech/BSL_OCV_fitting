figure(7)

data = mol.dQ_data;
subplot(1,2,1)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet

data = eol.dQ_data;
subplot(1,2,2)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet

set(gcf,'Position',[100 100 1400 550])


%%
figure(8)

data = mol.dQ_LLI;
subplot(1,2,1)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet


data = eol.dQ_LLI;
subplot(1,2,2)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet


set(gcf,'Position',[100 100 1400 550])


%%
figure(9)

data = mol.dQ_LAMp;
subplot(1,2,1)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet


data = eol.dQ_LAMp;
subplot(1,2,2)
imAlpha=ones(size(data));
imAlpha(isnan(data))=0;
imagesc(data,'AlphaData',imAlpha);
xticks([1 2 3 4]);xticklabels({'3.9V','4.0V','4.1V','4.2V'})
yticks([1 2 3 4 5]);yticklabels(flip({'0.5C','1C','2C','3C','4C'}))
set(gca,'color',[1 1 1]);
colorbar; clim([0 max(max(data))]); colormap jet


set(gcf,'Position',[100 100 1400 550])